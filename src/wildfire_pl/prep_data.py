from shapely.geometry import box, mapping
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask


def aoi_box(minx, miny, maxx, maxy):
    return box(minx, miny, maxx, maxy)


def size_shape_filter(
    crowns: gpd.GeoDataFrame, preds_df, resx, resy, area_m2=(20, 600), aspect=(0.4, 2.5)
):
    """Keep boxes by physical area (m²) and aspect ratio."""
    pix_area_m2 = abs(resx * resy)
    wpx = (preds_df.xmax - preds_df.xmin).astype(float)
    hpx = (preds_df.ymax - preds_df.ymin).astype(float)
    area = (wpx * hpx) * pix_area_m2
    hpx_safe = hpx.replace(0, np.nan)
    ar = (wpx / hpx_safe).astype(float)
    ar = ar.where(ar >= 1, 1 / ar)
    keep = area.between(*area_m2) & ar.between(*aspect)
    return crowns.loc[keep].reset_index(drop=True)


def ndvi_gate(crowns: gpd.GeoDataFrame, tif_path, ndvi_min=0.25):
    """Compute mean NDVI per polygon using NAIP NIR/Red and keep >= threshold."""
    with rasterio.open(tif_path) as src:
        red = src.read(1).astype(float)
        nir = src.read(4).astype(float)
        transform = src.transform
    ndvi = (nir - red) / (nir + red + 1e-6)
    vals = []
    for g in crowns.geometry:
        mask = geometry_mask(
            [mapping(g)], transform=transform, out_shape=ndvi.shape, invert=True
        )
        arr = ndvi[mask]
        vals.append(float(np.nanmean(arr)) if arr.size else np.nan)
    crowns = crowns.copy()
    crowns["mean_ndvi"] = vals
    return crowns[crowns.mean_ndvi >= ndvi_min].reset_index(drop=True)
