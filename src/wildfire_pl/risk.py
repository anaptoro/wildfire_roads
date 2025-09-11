# src/wildfire_pl/risk.py
from pathlib import Path
import os, sys, math, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize, geometry_mask
from rasterio.windows import Window, bounds as win_bounds, transform as win_transform
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString, box
from shapely.ops import substring
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
import folium
import branca.colormap as cm
from shapely.geometry import mapping as shp_mapping


# ---------------------------
# Utilities
# ---------------------------
def _stretch_rgb(arr, p=(2, 98)):
    arr = arr.astype(np.float32)
    out = np.zeros_like(arr, dtype=np.uint8)
    for i in range(arr.shape[-1]):
        lo, hi = np.percentile(arr[..., i], p)
        if hi <= lo:
            lo, hi = arr[..., i].min(), arr[..., i].max()
        ch = (np.clip((arr[..., i] - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(
            np.uint8
        )
        out[..., i] = ch
    return out


def _split_line_to_segments(line: LineString, seg_len_m: float) -> list[LineString]:
    if not isinstance(line, LineString) or line.length <= 0:
        return []
    d = 0.0
    out: list[LineString] = []
    while d < line.length:
        start, end = d, min(d + seg_len_m, line.length)
        seg = substring(line, start, end, normalized=False)
        if isinstance(seg, LineString) and seg.length > 0:
            out.append(seg)
        d = end
    return out


def _explode_to_spans(
    lines_gdf: gpd.GeoDataFrame, seg_len_m: float
) -> gpd.GeoDataFrame:
    rows = []
    for idx, row in lines_gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, LineString):
            parts = [geom]
        elif isinstance(geom, MultiLineString):
            parts = list(geom.geoms)
        else:
            parts = []
        for i, part in enumerate(parts):
            for j, seg in enumerate(_split_line_to_segments(part, seg_len_m)):
                rows.append(
                    {
                        "span_id": f"{idx}_{i}_{j}",
                        "geometry": seg,
                        "span_length_m": float(seg.length),
                    }
                )
    return gpd.GeoDataFrame(rows, crs=lines_gdf.crs)


def _poly_mean(arr: np.ndarray, transform, geom):
    mask = geometry_mask(
        [shp_mapping(geom)], transform=transform, out_shape=arr.shape, invert=True
    )
    vals = arr[mask]
    return float(np.nanmean(vals)) if vals.size else np.nan


# ---------------------------
# Worker (MUST be module-scope to pickle on macOS/Windows)
# ---------------------------
def _tile_reduce(args):
    import numpy as _np
    import rasterio as _rio
    from rasterio.features import rasterize as _rz
    from rasterio.windows import Window as _Window

    row_off = args["row_off"]
    col_off = args["col_off"]
    h = args["h"]
    w = args["w"]
    win_transform = args["win_transform"]
    shapes = args["shapes"]  # list of (geojson_mapping, label int)
    risk_source = args["risk_source"]

    # labels raster for this tile
    labels_tile = _rz(
        shapes, out_shape=(h, w), transform=win_transform, fill=0, dtype="int32"
    )

    # risk window
    if risk_source["mode"] == "tif":
        with _rio.open(risk_source["path"]) as rs:
            risk_win = rs.read(1, window=_Window(col_off, row_off, w, h)).astype(
                _np.float32
            )
    else:
        mm = _np.load(risk_source["path"], mmap_mode="r")
        risk_win = mm[row_off : row_off + h, col_off : col_off + w]

    labs = labels_tile.ravel()
    m = labs > 0
    if not m.any():
        return (
            _np.empty(0, _np.int32),
            _np.empty(0, _np.float64),
            _np.empty(0, _np.int64),
        )

    labs = labs[m]
    vals = risk_win.ravel()[m]

    uniq = _np.unique(labs)
    sums = _np.zeros_like(uniq, dtype=_np.float64)
    cnts = _np.zeros_like(uniq, dtype=_np.int64)

    lut = _np.empty(uniq.max() + 1, dtype=_np.int32)
    lut.fill(-1)
    lut[uniq] = _np.arange(len(uniq), dtype=_np.int32)

    idxs = lut[labs]
    _np.add.at(sums, idxs, vals)
    _np.add.at(cnts, idxs, 1)

    return uniq.astype(_np.int32), sums, cnts


# ---------------------------
# Risk map (raster + overlay)
# ---------------------------
def make_risk_maps(
    raster_tif: Path,
    roads_gpkg: Path,
    crowns_gdf: gpd.GeoDataFrame,
    out_risk_tif: Path,
    out_risk_png: Path,
    *,
    buf_for_lines_m: float = 1.0,
    decay_half_m: float = 20.0,
    crown_blur_px: int = 3,
    crown_weight: float = 0.6,
):
    """Create a risk raster (0..1) and a PNG overlay."""
    with rasterio.open(raster_tif) as src:
        H, W = src.height, src.width
        transform, crs = src.transform, src.crs
        resx, resy = src.res
        pix_m = float((abs(resx) + abs(resy)) / 2.0)
        rgb = np.moveaxis(src.read([1, 2, 3]), 0, 2)

    roads = gpd.read_file(roads_gpkg, layer="lines").to_crs(crs)
    buf = max(buf_for_lines_m, pix_m)
    roads_buf = roads.buffer(buf)
    line_mask = rasterize(
        [(geom, 1) for geom in roads_buf.geometry if geom and not geom.is_empty],
        out_shape=(H, W),
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    inv = (1 - line_mask).astype(np.uint8) * 255
    dist_px = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    dist_m = dist_px * pix_m
    k = np.log(2.0) / max(decay_half_m, 1e-6)
    proximity = np.exp(-k * dist_m)

    crowns_r = crowns_gdf.to_crs(crs)
    crown_mask = rasterize(
        [(g, 1) for g in crowns_r.geometry if g and not g.is_empty],
        out_shape=(H, W),
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(np.float32)

    if crown_blur_px > 0:
        crown_density = cv2.GaussianBlur(crown_mask, (0, 0), crown_blur_px)
        if crown_density.max() > 0:
            crown_density /= crown_density.max()
    else:
        crown_density = crown_mask

    risk = (1 - crown_weight) * proximity + crown_weight * (proximity * crown_density)
    risk = np.clip(risk, 0, 1).astype(np.float32)

    out_risk_tif = Path(out_risk_tif)
    with rasterio.open(
        out_risk_tif,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(risk, 1)

    rgb8 = _stretch_rgb(rgb)
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb8)
    plt.imshow(risk, cmap="inferno", alpha=0.4)
    plt.axis("off")
    plt.tight_layout()
    out_risk_png = Path(out_risk_png)
    out_risk_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_risk_png, dpi=200)
    plt.close()

    return str(out_risk_tif), str(out_risk_png)


# ---------------------------
# Risk spans (parallel)
# ---------------------------
def risk_spans_from_raster(
    raster_tif,
    roads_gpkg,
    crowns_gdf: gpd.GeoDataFrame,
    out_gpkg,
    *,
    risk_tif=None,
    seg_len_m=50.0,
    buf_m=15.0,
    n_workers: int | None = None,
):
    def log(msg: str):
        print(msg, flush=True, file=sys.stdout)

    # Compatibility helper: return **indices** of STRtree hits for both Shapely 1.8.x and 2.x
    def _query_idxs(
        tree: STRtree, geom, backing_geoms: list[BaseGeometry]
    ) -> np.ndarray:
        try:
            idxs = tree.query(geom, predicate="intersects")
            return np.asarray(idxs, dtype=int)
        except TypeError:
            cands = tree.query(geom)  # returns geometries in 1.8
            id_map = {id(g): i for i, g in enumerate(backing_geoms)}
            out = []
            for g in cands:
                i = id_map.get(id(g))
                if i is None:
                    continue
                try:
                    if geom.intersects(g):
                        out.append(i)
                except Exception:
                    out.append(i)
            return np.asarray(out, dtype=int)

    # --- base raster ---
    with rasterio.open(raster_tif) as base:
        base_crs = base.crs
        base_transform = base.transform
        H, W = base.height, base.width
        resx, resy = base.res
    pix_m = float((abs(resx) + abs(resy)) / 2.0)

    # --- lines to spans ---
    lines = gpd.read_file(roads_gpkg, layer="lines").to_crs(base_crs)
    if lines.empty:
        raise RuntimeError("No lines in 'lines' layer.")
    spans = _explode_to_spans(lines, seg_len_m=seg_len_m)
    if spans.empty:
        out_gpkg = Path(out_gpkg)
        out_gpkg.parent.mkdir(parents=True, exist_ok=True)
        spans.to_file(out_gpkg, layer="risk_spans", driver="GPKG")
        return spans, str(out_gpkg)

    spans = spans.reset_index(drop=True).copy()
    spans["_lbl"] = np.arange(1, len(spans) + 1, dtype=np.int32)

    span_buf_w = max(buf_m, pix_m)
    spans_buf = spans[["_lbl", "geometry"]].copy()
    spans_buf["geometry"] = spans.geometry.buffer(span_buf_w)
    buf_geoms: list[BaseGeometry] = list(spans_buf.geometry)
    labels = spans["_lbl"].to_numpy(dtype=np.int32)
    N = len(buf_geoms)

    # --- risk source (tif or memmapped npy) ---
    risk_source = {"mode": None, "path": None}
    if risk_tif is not None:
        risk_source.update(mode="tif", path=str(risk_tif))
    else:
        buf_lines = lines.buffer(max(1.0, pix_m))
        line_mask = rasterize(
            [(g, 1) for g in buf_lines.geometry if g and not g.is_empty],
            out_shape=(H, W),
            transform=base_transform,
            fill=0,
            dtype="uint8",
        )
        inv = (1 - line_mask).astype(np.uint8) * 255
        dist_px = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        dist_m = dist_px * pix_m
        k = np.log(2.0) / 20.0
        risk = np.exp(-k * dist_m).astype(np.float32)

        tmp_dir = tempfile.gettempdir()
        risk_npy = os.path.join(tmp_dir, f"_wf_risk_{os.getpid()}.npy")
        np.save(risk_npy, risk)
        del risk
        risk_source.update(mode="npy", path=risk_npy)

    # --- build tile tasks (use STRtree indices; pass mapped geoms) ---
    tree = STRtree(buf_geoms)
    TILE = 2048 if max(H, W) > 6000 else 4096
    tiles = []
    for row_off in range(0, H, TILE):
        for col_off in range(0, W, TILE):
            h = min(TILE, H - row_off)
            w = min(TILE, W - col_off)
            win = Window(col_off, row_off, w, h)
            wb = win_bounds(win, base_transform)  # (left, bottom, right, top)
            wpoly = box(*wb)

            idxs = _query_idxs(tree, wpoly, buf_geoms)
            if idxs.size == 0:
                continue

            shapes = []
            for i in idxs:
                g = buf_geoms[int(i)]
                if isinstance(g, BaseGeometry) and (g is not None) and (not g.is_empty):
                    shapes.append(
                        (shp_mapping(g), int(labels[int(i)]))
                    )  # mapped geom → picklable
            if shapes:
                tiles.append(
                    {
                        "row_off": row_off,
                        "col_off": col_off,
                        "h": h,
                        "w": w,
                        "win_transform": win_transform(win, base_transform),
                        "shapes": shapes,
                    }
                )

    log(
        f"[risk] size: {W}x{H}px (~{(W * H) / 1e6:.1f} MP)  spans: {N}  tiles(with work): {len(tiles)}"
    )

    # --- parallel over tiles ---
    sums = np.zeros(N + 1, dtype=np.float64)
    cnts = np.zeros(N + 1, dtype=np.int64)
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    if tiles:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [
                ex.submit(_tile_reduce, {**t, "risk_source": risk_source})
                for t in tiles
            ]
            for i, fut in enumerate(as_completed(futs), 1):
                uniq, s, c = fut.result()
                if uniq.size:
                    sums[uniq] += s
                    cnts[uniq] += c
                if i % 8 == 0 or i == len(futs):
                    log(f"[risk] tiles done: {i}/{len(futs)}")
    else:
        log("[risk] no tiles intersected buffers; all means = 0")

    means = np.divide(
        sums[1:], cnts[1:], out=np.zeros(N, dtype=np.float32), where=cnts[1:] > 0
    ).astype(np.float32)
    spans["mean_risk"] = means

    # --- crowns per span ---
    crowns_near = np.zeros(N, dtype=np.int32)
    if len(crowns_gdf):
        crowns_proj = crowns_gdf.to_crs(base_crs)
        crowns_pts = crowns_proj.geometry.centroid
        tree_spans = STRtree(buf_geoms)
        # best effort: indices if available; else map back
        try:
            pairs = tree_spans.query(crowns_pts, predicate="covers")
            rows = []
            # Normalize possible returns
            if (
                isinstance(pairs, np.ndarray)
                and pairs.ndim == 2
                and pairs.shape[0] == 2
            ):
                pass  # already pairs
            else:
                for p_i, idxs in enumerate(pairs):
                    for s_i in np.asarray(idxs, dtype=int):
                        rows.append((p_i, int(s_i)))
                pairs = (
                    np.asarray(rows, dtype=int).T
                    if rows
                    else np.empty((2, 0), dtype=int)
                )
        except Exception:
            rows = []
            id_map = {id(g): i for i, g in enumerate(buf_geoms)}
            for pi, pt in enumerate(crowns_pts):
                cands = tree_spans.query(pt)
                for g in cands:
                    i = id_map.get(id(g))
                    if i is not None and pt.intersects(g):
                        rows.append((pi, int(i)))
            pairs = (
                np.asarray(rows, dtype=int).T if rows else np.empty((2, 0), dtype=int)
            )

        if pairs.size:
            counts = np.bincount(pairs[1], minlength=N)
            crowns_near = counts.astype(np.int32)

    spans["crowns_near"] = crowns_near

    # --- ranking ---
    spans["crowns_per_100m"] = (
        spans["crowns_near"] / (spans["span_length_m"].clip(lower=1) / 100.0)
    ).astype("float32")
    spans["risk_score"] = (
        spans["mean_risk"] * (1.0 + (spans["crowns_per_100m"] / 5.0))
    ).astype("float32")
    spans["risk_score"].replace([np.inf, -np.inf], 0.0, inplace=True)

    spans.sort_values(
        ["risk_score", "crowns_near"], ascending=[False, False], inplace=True
    )
    spans.reset_index(drop=True, inplace=True)

    # --- save ---
    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    spans.to_file(out_gpkg, layer="risk_spans", driver="GPKG")
    log(f"[risk] wrote spans → {out_gpkg}")

    if risk_source["mode"] == "npy":
        try:
            os.remove(risk_source["path"])
        except Exception:
            pass

    return spans, str(out_gpkg)


# ---------------------------
# Folium map for risk spans
# ---------------------------
def folium_risk_map(aoi_bbox, spans_gdf: gpd.GeoDataFrame, out_html):
    minx, miny, maxx, maxy = aoi_bbox
    m = folium.Map(
        location=[(miny + maxy) / 2, (minx + maxx) / 2],
        zoom_start=14,
        control_scale=True,
    )
    folium.Rectangle(
        bounds=[(miny, minx), (maxy, maxx)], color="blue", weight=2, fill=False
    ).add_to(m)

    scores = spans_gdf.get("risk_score")
    if scores is not None and len(scores):
        vmax = float(np.nanmax(np.asarray(scores, dtype=float)))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
    else:
        vmax = 1.0

    cmap = cm.LinearColormap(
        ["#2b83ba", "#abdda4", "#ffffbf", "#fdae61", "#d7191c"], vmin=0, vmax=vmax
    )
    cmap.caption = "Risk score"

    def style_fn(feat):
        val = feat["properties"].get("risk_score", 0.0)
        try:
            s = float(val)
        except (TypeError, ValueError):
            s = 0.0
        if not np.isfinite(s):
            s = 0.0
        return {"color": cmap(s), "weight": 4, "opacity": 0.9}

    folium.GeoJson(
        spans_gdf.to_crs(4326).__geo_interface__,
        style_function=style_fn,
        name="Risk spans",
    ).add_to(m)
    cmap.add_to(m)
    folium.LayerControl().add_to(m)

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    return str(out_html)
