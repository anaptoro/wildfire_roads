# tests/test_prep_data.py
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.transform import from_origin
from rasterio.io import MemoryFile

from wildfire_pl.prep_data import size_shape_filter, ndvi_gate


def _tiny_rgba_raster(width=20, height=20, res=1.0):
    transform = from_origin(0, height, res, res)
    data = np.zeros((4, height, width), dtype=np.uint8)

    data[0] = 50  # R
    data[1] = 50  # G
    data[2] = 50  # B
    data[3] = 200  # NIR
    mem = MemoryFile()
    with mem.open(
        driver="GTiff",
        height=height,
        width=width,
        count=4,
        dtype="uint8",
        transform=transform,
        crs="EPSG:3857",
    ) as dst:
        dst.write(data)
    return mem


def test_size_shape_filter():
    gdf = gpd.GeoDataFrame(geometry=[box(1, 1, 2, 2), box(1, 1, 6, 6)], crs="EPSG:3857")

    import pandas as pd

    preds_df = pd.DataFrame(
        {"xmin": [1, 1], "ymin": [1, 1], "xmax": [2, 6], "ymax": [2, 6]}
    )
    # resx,resy=1 => pixel area = 1 m² for test
    kept = size_shape_filter(
        gdf, preds_df, resx=1.0, resy=1.0, area_m2=(10, 1000), aspect=(0.4, 2.5)
    )
    # only the larger polygon should remain
    assert len(kept) == 1
    assert kept.geometry.iloc[0].bounds == (1.0, 1.0, 6.0, 6.0)


def test_ndvi_gate_keeps_green(tmp_path):
    mem = _tiny_rgba_raster()
    tif = tmp_path / "tiny.tif"
    with mem.open() as src:
        profile = src.profile
        data = src.read()
    with rasterio.open(tif, "w", **profile) as dst:
        dst.write(data)

    gdf = gpd.GeoDataFrame(geometry=[box(5, 5, 10, 10)], crs="EPSG:3857")
    out = ndvi_gate(gdf, tif, ndvi_min=0.1)
    assert len(out) == 1
