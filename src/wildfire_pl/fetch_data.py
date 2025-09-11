from pathlib import Path
import time
import random
from shapely.geometry import box, shape, mapping
from shapely.ops import transform as shp_transform
from pystac_client import Client
from pystac_client.exceptions import APIError
import planetary_computer as pc
import rasterio
import rasterio.mask
from rasterio.warp import transform_geom, transform_bounds
from pyproj import Transformer
from typing import Any, Dict
from collections.abc import Iterable
import requests
import geopandas as gpd
from shapely.geometry import LineString


def search_naip(aoi_bbox, limit=10):
    minx, miny, maxx, maxy = aoi_bbox
    AOI = box(minx, miny, maxx, maxy)
    stac = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = stac.search(
        collections=["naip"],
        intersects=mapping(AOI),
        sortby=[{"field": "datetime", "direction": "desc"}],
        limit=limit,
    )
    return search, AOI


def _clip_one(href, AOI_wgs84, out_tif):
    """Robust clip using rasterio.warp instead of shapely.buffer()."""
    with rasterio.open(href) as src:
        # 1) Quick reject in WGS84: project raster bounds -> 4326 and test against AOI (also 4326)
        rb_wgs84 = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        if not box(*rb_wgs84).intersects(AOI_wgs84):
            return False

        # 2) Reproject the AOI geometry into the raster CRS (GDAL/PROJ does it; no shapely buffer)
        aoi_r_geojson = transform_geom(
            "EPSG:4326", src.crs.to_string(), mapping(AOI_wgs84), precision=6
        )
        aoi_r = shape(aoi_r_geojson)

        # 3) If numeric jitter still misses, expand the AOI **bounds** by one pixel (in raster units)
        if not aoi_r.intersects(box(*src.bounds)):
            pix = float(max(abs(src.res[0]), abs(src.res[1])))
            x0, y0, x1, y1 = aoi_r.bounds
            aoi_r = box(x0 - pix, y0 - pix, x1 + pix, y1 + pix)
            aoi_r_geojson = mapping(aoi_r)
            if not aoi_r.intersects(box(*src.bounds)):
                return False

        # 4) Clip
        img, T = rasterio.mask.mask(src, [aoi_r_geojson], crop=True)
        meta = src.meta.copy()
        meta.update(height=img.shape[1], width=img.shape[2], transform=T)

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **meta) as dst:
        dst.write(img)
    return True


def fetch_and_clip_naip(aoi_bbox, out_dir, max_tries=5, base_sleep=1.5):
    """Find a NAIP item that intersects AOI and write clipped TIFF. Retries on STAC 5xx errors."""
    out_dir = Path(out_dir)
    out_tif = out_dir / "naip_aoi.tif"
    # skip if we already have it
    if out_tif.exists():
        return out_tif

    search, AOI = search_naip(aoi_bbox, limit=10)
    last_err = None
    for attempt in range(1, max_tries + 1):
        try:
            # NOTE: use .items() (get_items is deprecated)
            for item in search.items():
                href = pc.sign(item).assets["image"].href
                if _clip_one(href, AOI, out_tif):
                    return out_tif
            raise RuntimeError("No NAIP item intersected the AOI.")
        except APIError as e:
            last_err = e
            # Retry only for server-side problems (5xx)
            status = getattr(e, "status_code", None)
            if status is not None and 500 <= status < 600:
                sleep = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                print(
                    f"[NAIP] STAC {status} on attempt {attempt}/{max_tries}; retrying in {sleep:.1f}s..."
                )
                time.sleep(sleep)
                continue
            # non-retryable error
            raise
    # exhausted retries
    raise RuntimeError(
        f"Planetary Computer STAC failed after {max_tries} tries: {last_err}"
    )


DEFAULT_HIGHWAY_CLASSES = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
]


def fetch_highways(
    aoi_bbox: tuple[float, float, float, float],
    raster_tif: str | Path,
    out_gpkg: str | Path,
    classes: Iterable[str] | None = None,
    buffer_m: float = 30.0,
) -> Path:
    """Fetch OSM highways in bbox, reproject to raster CRS, buffer, and save GPKG (layers: lines, lines_buffer)."""
    classes = list(classes or DEFAULT_HIGHWAY_CLASSES)

    # Overpass bbox order: south, west, north, east
    minx, miny, maxx, maxy = aoi_bbox
    south, west, north, east = miny, minx, maxy, maxx
    regex = "^(" + "|".join(classes) + ")$"

    q = f"""
    [out:json][timeout:60];
    (
      way["highway"~"{regex}"]({south},{west},{north},{east});
    );
    out tags geom;
    """

    resp = requests.post("https://overpass-api.de/api/interpreter", data={"data": q})
    resp.raise_for_status()
    data: Dict[str, Any] = resp.json()

    feats: list[dict[str, Any]] = []
    for el in data.get("elements", []):
        if el.get("type") == "way" and "geometry" in el:
            coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
            if len(coords) >= 2:
                feats.append(
                    {
                        "geometry": LineString(coords),
                        "highway": el.get("tags", {}).get("highway"),
                    }
                )

    roads = gpd.GeoDataFrame(feats, crs=4326)
    if roads.empty:
        raise RuntimeError("No OSM highways found in AOI.")

    # Reproject to raster CRS and buffer in meters
    with rasterio.open(raster_tif) as src:
        roads_proj = roads.to_crs(src.crs)

    roads_buf = roads_proj.copy()
    roads_buf["geometry"] = roads_proj.geometry.buffer(buffer_m)

    # Ensure proper path object and create parent dir
    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    # Write layers
    roads_proj.to_file(out_gpkg, layer="lines", driver="GPKG")
    roads_buf.to_file(out_gpkg, layer="lines_buffer", driver="GPKG")
    return out_gpkg
