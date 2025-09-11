from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.plot import reshape_as_image

# Force non-interactive backend for containers/servers
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def overlay_png(
    raster_tif: Path,
    crowns_gdf: gpd.GeoDataFrame,
    out_png: Path,
    *,
    alpha: float = 0.35,
    line_w: float = 0.8,
    dpi: int = 200,
) -> str:
    """Save an RGB PNG overlay with crown polygons outlined."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(raster_tif) as src:
        rgb = src.read([1, 2, 3])
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb_img = reshape_as_image(rgb)
        crs = src.crs

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    ax.imshow(rgb_img)
    if len(crowns_gdf):
        cg = crowns_gdf.to_crs(crs)
        # draw outlines so it’s readable over imagery
        cg.boundary.plot(ax=ax, linewidth=line_w, color="yellow", alpha=0.9)
    ax.set_axis_off()
    fig.tight_layout(pad=0)

    tmp = out_png.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    tmp.rename(out_png)
    return str(out_png)
