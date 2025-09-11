import numpy as np
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.transform import from_origin
from rasterio.io import MemoryFile
import matplotlib.pyplot as plt
import os
import tempfile
import pytest
from pathlib import Path
import folium
import branca.colormap as cm
from rasterio.features import rasterize
from shapely.geometry import LineString, MultiLineString, mapping
from shapely.ops import substring

from wildfire_pl.risk import make_risk_maps, risk_spans_from_raster


def _write_rgbn(path: Path, w=40, h=40, crs="EPSG:3857"):
    transform = from_origin(0, h, 1, 1)
    arr = np.zeros((4, h, w), dtype=np.uint8)
    arr[0] = 60  # R
    arr[1] = 60  # G
    arr[2] = 60  # B
    arr[3] = 180  # NIR -> positive NDVI
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=4,
        dtype="uint8",
        transform=transform,
        crs=crs,
    ) as dst:
        dst.write(arr)


def test_make_risk_and_spans(tmp_path):
    tif = tmp_path / "chip.tif"
    _write_rgbn(tif)

    # one straight road
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 20), (40, 20)])], crs="EPSG:3857"
    )
    roads_gpkg = tmp_path / "roads.gpkg"
    roads.to_file(roads_gpkg, layer="lines", driver="GPKG")
    roads.assign(geometry=roads.buffer(5)).to_file(
        roads_gpkg, layer="lines_buffer", driver="GPKG"
    )

    # two crown boxes near the line
    crowns = gpd.GeoDataFrame(
        geometry=[box(10, 15, 12, 17), box(28, 22, 30, 24)], crs="EPSG:3857"
    )

    risk_tif, risk_png = make_risk_maps(
        tif, roads_gpkg, crowns, tmp_path / "risk.tif", tmp_path / "risk.png"
    )
    assert Path(risk_tif).exists()
    assert Path(risk_png).exists()

    spans_gdf, spans_path = risk_spans_from_raster(
        tif, roads_gpkg, crowns, tmp_path / "spans.gpkg", risk_tif=risk_tif
    )
    assert Path(spans_path).exists()
    assert "risk_score" in spans_gdf.columns
