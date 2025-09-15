# src/wildfire_pl/fetch_data.py
from __future__ import annotations

from pathlib import Path
import time
import random
from typing import Any, Dict

import requests
import geopandas as gpd
import rasterio
import rasterio.mask
from rasterio.warp import transform_geom, transform_bounds
from shapely.geometry import box, shape, mapping, LineString
from collections.abc import Iterable

from pystac_client import Client
from pystac_client.exceptions import APIError
import planetary_computer as pc

# write GPKG via pyogrio (avoids Fiona/GDAL hangs in containers)
from pyogrio import write_dataframe


# -------------------------
# NAIP search + clipping
# -------------------------

def search_naip(aoi_bbox, limit: int = 10):
    """Return (search, AOI polygon) for NAIP over the given bbox (EPSG:4326)."""
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


def _clip_one(href: str, AOI_wgs84, out_tif: Path) -> bool:
    """Robust clip using rasterio.warp reproject (no shapely buffering in projected space)."""
    with rasterio.open(href) as src:
        # 1) Quick reject: reproject raster bounds → EPSG:4326 and test against AOI
        rb_wgs84 = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        if not box(*rb_wgs84).intersects(AOI_wgs84):
            return False

        # 2) Reproject AOI into raster CRS via GDAL/PROJ (precise, stable)
        aoi_r_geojson = transform_geom(
            "EPSG:4326", src.crs.to_string(), mapping(AOI_wgs84), precision=6
        )
        aoi_r = shape(aoi_r_geojson)

        # 3) If floating-point jitter misses by a hair, expand AOI by ~1 pixel in raster units
        if not aoi_r.intersects(box(*src.bounds)):
            pix = float(max(abs(src.res[0]), abs(src.res[1])))
            aoi_r = aoi_r.buffer(pix)           # preserve shape (vs expanding bbox)
            aoi_r_geojson = mapping(aoi_r)
            if not aoi_r.intersects(box(*src.bounds)):
                return False

        # 4) Clip and write
        img, T = rasterio.mask.mask(src, [aoi_r_geojson], crop=True)
        meta = src.meta.copy()
        meta.update(height=img.shape[1], width=img.shape[2], transform=T)

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **meta) as dst:
        dst.write(img)
    return True


def fetch_and_clip_naip(aoi_bbox, out_dir, max_tries: int = 5, base_sleep: float = 1.5) -> Path:
    """Find a NAIP item that intersects AOI and write clipped TIFF. Retries on STAC 5xx errors."""
    out_dir = Path(out_dir)
    out_tif = out_dir / "naip_aoi.tif"
    if out_tif.exists():
        return out_tif

    search, AOI = search_naip(aoi_bbox, limit=10)
    last_err = None
    for attempt in range(1, max_tries + 1):
        try:
            # Iterate most recent items first
            for item in search.items():
                href = pc.sign(item).assets["image"].href
                if _clip_one(href, AOI, out_tif):
                    return out_tif
            raise RuntimeError("No NAIP item intersected the AOI.")
        except APIError as e:
            last_err = e
            status = getattr(e, "status_code", None)
            if status is not None and 500 <= status < 600:
                sleep = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                print(f"[NAIP] STAC {status} on attempt {attempt}/{max_tries}; retrying in {sleep:.1f}s...")
                time.sleep(sleep)
                continue
            raise
    raise RuntimeError(f"Planetary Computer STAC failed after {max_tries} tries: {last_err}")


# -------------------------
# OSM highways via Overpass
# -------------------------

DEFAULT_HIGHWAY_CLASSES = [
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
]


def fetch_highways(
    aoi_bbox: tuple[float, float, float, float],
    raster_tif: str | Path,
    out_gpkg: str | Path,
    classes: Iterable[str] | None = None,
    buffer_m: float = 30.0,
) -> Path:
    """
    Fetch OSM highways in bbox, reproject to raster CRS, buffer, and save GPKG (layers: lines, lines_buffer).

    Uses pyogrio to write GPKG to avoid Fiona/GDAL hangs in containers.
    """
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

    # Robust request (no silent hangs), and friendly UA
    headers = {
        "User-Agent": "wildfire_pl/0.1 (contact: you@example.com)",
        "Accept": "application/json",
    }
    resp = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": q},
        headers=headers,
        timeout=(10, 120),   # 10s connect, 120s read
    )
    resp.raise_for_status()
    data: Dict[str, Any] = resp.json()

    feats: list[dict[str, Any]] = []
    for el in data.get("elements", []):
        if el.get("type") == "way" and "geometry" in el:
            coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
            if len(coords) >= 2:
                feats.append({
                    "geometry": LineString(coords),
                    "highway": el.get("tags", {}).get("highway"),
                })

    roads = gpd.GeoDataFrame(feats, crs=4326)
    if roads.empty:
        raise RuntimeError("No OSM highways found in AOI.")

    # Reproject to raster CRS and buffer in meters
    with rasterio.open(raster_tif) as src:
        dst_crs = src.crs
    roads_proj = roads.to_crs(dst_crs)

    roads_buf = roads_proj.copy()
    roads_buf["geometry"] = roads_proj.geometry.buffer(buffer_m)

    # Ensure path exists
    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    # Write using pyogrio (fast, reliable in containers)
    write_dataframe(roads_proj, out_gpkg, layer="lines", driver="GPKG")
    write_dataframe(roads_buf, out_gpkg, layer="lines_buffer", driver="GPKG")

    return out_gpkg