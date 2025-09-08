# src/wildfire_pl/risk.py
import numpy as np
from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString, mapping
from shapely.ops import substring
from rasterio.features import geometry_mask
import folium
import branca.colormap as cm


def _stretch_rgb(arr, p=(2, 98)):
    arr = arr.astype(np.float32)
    out = np.zeros_like(arr, dtype=np.uint8)
    for i in range(arr.shape[-1]):
        lo, hi = np.percentile(arr[..., i], p)
        if hi <= lo:
            lo, hi = arr[..., i].min(), arr[..., i].max()
        ch = (np.clip((arr[..., i] - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
        out[..., i] = ch
    return out

def make_risk_maps(
    raster_tif: Path,
    roads_gpkg: Path,
    crowns_gdf: gpd.GeoDataFrame,
    out_risk_tif: Path,
    out_risk_png: Path,
    *,
    buf_for_lines_m: float = 1.0,     # widen roads a tad when rasterizing
    decay_half_m: float = 20.0,       # distance half-life (m) for proximity risk
    crown_blur_px: int = 3,           # smooth crown density
    crown_weight: float = 0.6,        # blend factor for proximity vs crowns
):
    """Create a risk raster (0..1) and a PNG overlay.

    risk = (1 - crown_weight) * proximity_to_roads + crown_weight * (proximity * crown_density)
    """
    with rasterio.open(raster_tif) as src:
        H, W = src.height, src.width
        transform, crs = src.transform, src.crs
        resx, resy = src.res
        pix_m = float((abs(resx) + abs(resy)) / 2.0)
        rgb = np.moveaxis(src.read([1, 2, 3]), 0, 2)

    roads = gpd.read_file(roads_gpkg, layer="lines").to_crs(crs)
    # rasterize (buffer a touch so thin lines are captured)
    buf = max(buf_for_lines_m, pix_m)
    roads_buf = roads.buffer(buf)
    line_mask = rasterize(
        [(geom, 1) for geom in roads_buf.geometry if geom and not geom.is_empty],
        out_shape=(H, W),
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    # Distance to nearest road (in meters) using OpenCV EDT
    inv = (1 - line_mask).astype(np.uint8) * 255     # 255 where NOT road
    dist_px = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    dist_m = dist_px * pix_m
    k = np.log(2.0) / max(decay_half_m, 1e-6)        # half-life in meters
    proximity = np.exp(-k * dist_m)                  # 1 at road, ->0 far away

    # Crown density map
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

    # Save GeoTIFF
    out_risk_tif = Path(out_risk_tif)
    with rasterio.open(
        out_risk_tif, "w",
        driver="GTiff", height=H, width=W, count=1,
        dtype="float32", crs=crs, transform=transform
    ) as dst:
        dst.write(risk, 1)

    # Save PNG overlay
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
def _split_line_to_segments(line: LineString, seg_len_m: float):
    if not isinstance(line, LineString) or line.length <= 0:
        return []
    d = 0.0
    out = []
    while d < line.length:
        start, end = d, min(d + seg_len_m, line.length)
        seg = substring(line, start, end, normalized=False)
        if isinstance(seg, LineString) and seg.length > 0:
            out.append(seg)
        d = end
    return out

def _explode_to_spans(lines_gdf: gpd.GeoDataFrame, seg_len_m: float) -> gpd.GeoDataFrame:
    rows = []
    for idx, row in lines_gdf.iterrows():
        geom = row.geometry
        parts = [geom] if isinstance(geom, LineString) else list(geom.geoms) if isinstance(geom, MultiLineString) else []
        for part in parts:
            for i, seg in enumerate(_split_line_to_segments(part, seg_len_m)):
                rows.append({"span_id": f"{idx}_{i}", "geometry": seg, "span_length_m": seg.length})
    return gpd.GeoDataFrame(rows, crs=lines_gdf.crs)

# --- mean of a raster under a polygon ---
def _poly_mean(arr, transform, geom):
    mask = geometry_mask([mapping(geom)], transform=transform, out_shape=arr.shape, invert=True)
    vals = arr[mask]
    return float(np.nanmean(vals)) if vals.size else np.nan

def risk_spans_from_raster(
    raster_tif,            # the NAIP tif (for CRS/transform)
    roads_gpkg,            # GPKG with 'lines' layer (roads or powerlines)
    crowns_gdf: gpd.GeoDataFrame,
    out_gpkg,
    *,
    risk_tif=None,         # optional: path to precomputed risk.tif; if None, compute proximity-only
    seg_len_m=50.0,
    buf_m=15.0,
):
    """Build span segments, compute mean risk per span buffer, add crown density stats, save GPKG."""
    # open base raster (to sync CRS/transform/res)
    with rasterio.open(raster_tif) as base:
        base_crs = base.crs
        base_transform = base.transform
        H, W = base.height, base.width
        resx, resy = base.res
        pix_m = float((abs(resx) + abs(resy)) / 2.0)

    # get roads/powerlines
    lines = gpd.read_file(roads_gpkg, layer="lines").to_crs(base_crs)
    spans = _explode_to_spans(lines, seg_len_m=seg_len_m)

    # span buffers used for both risk sampling and crown counting
    spans_buf = spans.copy()
    spans_buf["geometry"] = spans_buf.geometry.buffer(max(buf_m, pix_m))

    # risk array: either supplied risk.tif or distance-to-lines → proximity
    if risk_tif is not None:
        with rasterio.open(risk_tif) as rs:
            assert rs.crs == base_crs, "risk.tif CRS must match base raster"
            risk = rs.read(1).astype(np.float32)
            r_transform = rs.transform
    else:
        # build proximity risk on-the-fly (like in make_risk_maps)
        inv = np.ones((H, W), dtype=np.uint8)
        buf_lines = lines.buffer(max(1.0, pix_m))
        line_mask = rasterize([(g, 1) for g in buf_lines.geometry if g and not g.is_empty],
                              out_shape=(H, W), transform=base_transform, fill=0, dtype="uint8")
        inv[line_mask == 1] = 0
        dist_px = cv2.distanceTransform((inv * 255).astype(np.uint8), cv2.DIST_L2, 3)
        dist_m = dist_px * pix_m
        k = np.log(2.0) / 20.0  # half-life 20 m
        risk = np.exp(-k * dist_m).astype(np.float32)
        r_transform = base_transform

    # mean risk per span buffer
    spans["mean_risk"] = [
        _poly_mean(risk, r_transform, geom) for geom in spans_buf.geometry
    ]

    # crown counts per span
    crowns_proj = crowns_gdf.to_crs(base_crs).copy()
    centroids = crowns_proj.geometry.centroid
    crowns_pts = crowns_proj.copy() 
    crowns_pts["geometry"] = centroids

    joined = gpd.sjoin(crowns_pts[["geometry"]], spans_buf[["span_id","geometry"]],
                       predicate="within", how="left")
    counts = (joined.dropna(subset=["span_id"])
                    .groupby("span_id").size()
                    .reset_index(name="crowns_near"))
    spans = spans.merge(counts, on="span_id", how="left").fillna({"crowns_near": 0})

    # --- add these lines ---
    spans["mean_risk"] = spans["mean_risk"].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    spans["crowns_near"] = spans["crowns_near"].astype(int)
    spans["crowns_per_100m"] = (spans["crowns_near"] / (spans["span_length_m"].clip(lower=1) / 100.0)).astype("float32")

    spans["risk_score"] = (spans["mean_risk"] * (1.0 + (spans["crowns_per_100m"] / 5.0))).astype("float32")
    spans["risk_score"] = spans["risk_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    spans = spans.sort_values(["risk_score","crowns_near"], ascending=[False, False]).reset_index(drop=True)

    # save
    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    spans.to_file(out_gpkg, layer="risk_spans", driver="GPKG")
    return spans, str(out_gpkg)

def folium_risk_map(aoi_bbox, spans_gdf: gpd.GeoDataFrame, out_html):
    minx, miny, maxx, maxy = aoi_bbox
    m = folium.Map(location=[(miny+maxy)/2, (minx+maxx)/2], zoom_start=14, control_scale=True)
    folium.Rectangle(bounds=[(miny, minx), (maxy, maxx)], color="blue", weight=2, fill=False).add_to(m)

    scores = spans_gdf.get("risk_score")
    if scores is not None:
        vmax = float(np.nanmax(np.asarray(scores, dtype=float)))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
    else:
        vmax = 1.0

    cmap = cm.LinearColormap(["#2b83ba","#abdda4","#ffffbf","#fdae61","#d7191c"], vmin=0, vmax=vmax)
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

    folium.GeoJson(spans_gdf.to_crs(4326).__geo_interface__, style_function=style_fn, name="Risk spans").add_to(m)
    cmap.add_to(m)
    folium.LayerControl().add_to(m)

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    return str(out_html)
