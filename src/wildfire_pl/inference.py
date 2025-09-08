import numpy as np
import cv2
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon
from rasterio.transform import xy
from deepforest import main as df_main

def run_deepforest(rgb_array, patch=800, overlap=0.2, iou=0.4, conf=0.3, upscale=1.0):
    """Tiled inference on an RGB array (H,W,3). Returns a pandas DataFrame with box coords in pixel space."""
    m = df_main.deepforest()
    m.use_release()

    img = rgb_array
    if img.dtype != np.uint8:
        lo, hi = float(img.min()), float(img.max())
        img = ((img - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)

    if upscale != 1.0:
        h, w = img.shape[:2]
        img_up = cv2.resize(img, (int(w*upscale), int(h*upscale)), interpolation=cv2.INTER_CUBIC)
    else:
        img_up = img

    preds = m.predict_tile(image=img_up, patch_size=patch, patch_overlap=overlap,
                           iou_threshold=iou, return_plot=False)
    if len(preds):
        if "score" in preds: 
            preds = preds[preds.score >= conf].copy()
        if upscale != 1.0:
            for c in ("xmin","xmax","ymin","ymax"):
                preds[c] = preds[c] / upscale
        preds.reset_index(drop=True, inplace=True)
    return preds

def boxes_to_geoms(preds_df, transform, crs):
    polys=[]
    for r in preds_df.itertuples():
        corners = [(r.ymin,r.xmin),(r.ymin,r.xmax),(r.ymax,r.xmax),(r.ymax,r.xmin)]
        corners_xy = [xy(transform, rr, cc, offset="center") for rr,cc in corners]
        polys.append(Polygon(corners_xy))
    return gpd.GeoDataFrame(preds_df.copy(), geometry=polys, crs=crs)

def detect_on_tif(tif_path, patch=800, overlap=0.2, iou=0.4, conf=0.3, upscale=1.0):
    """Read RGB from NAIP TIFF, run DeepForest tiled inference, return (crowns_gdf, (resx,resy), preds_df)."""
    with rasterio.open(tif_path) as src:
        rgb = np.moveaxis(src.read([1,2,3]), 0, 2)
        transform, crs = src.transform, src.crs
        resx, resy = src.res
    preds = run_deepforest(rgb, patch=patch, overlap=overlap, iou=iou, conf=conf, upscale=upscale)
    crowns = boxes_to_geoms(preds, transform, crs)
    return crowns, (resx, resy), preds
