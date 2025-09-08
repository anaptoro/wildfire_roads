import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import plotting_extent
import folium

def _stretch(arr, p=(2,98)):
    out = np.zeros_like(arr, dtype=np.uint8)
    for i in range(arr.shape[-1]):
        lo, hi = np.percentile(arr[...,i], p)
        if hi <= lo: 
            lo, hi = arr[...,i].min(), arr[...,i].max()
        x = (arr[...,i]-lo)/(hi-lo+1e-6)
        out[...,i]=(np.clip(x,0,1)*255).astype(np.uint8)
    return out

def overlay_png(raster_tif, crowns_gdf, out_png):
    with rasterio.open(raster_tif) as src:
        arr = np.moveaxis(src.read([1,2,3]), 0, -1).astype(np.float32)
        extent = plotting_extent(src)
    rgb8 = _stretch(arr)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(rgb8, extent=extent)
    if len(crowns_gdf):
        crowns_gdf.boundary.plot(ax=ax, linewidth=0.8, edgecolor="yellow", alpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    return out_png

def lines_map(aoi_bbox, lines_gdf, out_html):
    minx,miny,maxx,maxy = aoi_bbox
    m = folium.Map(location=[(miny+maxy)/2,(minx+maxx)/2], zoom_start=14, control_scale=True)
    folium.Rectangle(bounds=[(miny,minx),(maxy,maxx)], color="blue", weight=2, fill=False).add_to(m)
    folium.GeoJson(
        data=lines_gdf.to_crs(4326).__geo_interface__,
        name="Lines",
        style_function=lambda f: {"color":"red","weight":2,"opacity":0.8}
    ).add_to(m)
    m.save(out_html)
    return out_html
