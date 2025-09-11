import json
from pathlib import Path
from typing import Tuple

import click

from .config import AOIS, DEFAULTS
from .fetch_data import fetch_and_clip_naip, fetch_highways
from .inference import detect_on_tif
from .prep_data import size_shape_filter, ndvi_gate
from .viz import overlay_png
from .risk import make_risk_maps, risk_spans_from_raster, folium_risk_map


def _echo(s: str, quiet: bool):
    if not quiet:
        click.echo(s)


@click.command(context_settings={"show_default": True})
@click.option(
    "--aoi",
    type=click.Choice(sorted(AOIS)),
    help="Named AOI from wildfire_pl.config.AOIS",
)
@click.option(
    "--bbox",
    type=click.Tuple([float, float, float, float]),
    help="Custom bbox as (minx,miny,maxx,maxy) in EPSG:4326",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default="outputs",
    help="Output directory",
)
# Inference knobs
@click.option("--upscale", type=float, default=DEFAULTS["UPSCALE"])
@click.option("--patch", type=int, default=DEFAULTS["PATCH"])
@click.option("--overlap", type=float, default=DEFAULTS["OVERLAP"])
@click.option("--iou", type=float, default=DEFAULTS["IOU"])
@click.option("--conf", type=float, default=DEFAULTS["CONF"])
# Prep / filters
@click.option("--ndvi-min", "ndvi_min", type=float, default=DEFAULTS["NDVI_MIN"])
@click.option(
    "--area-m2",
    "area_m2",
    type=click.Tuple([float, float]),
    default=DEFAULTS["AREA_M2"],
    help="Min,max polygon area in m²",
)
@click.option(
    "--aspect",
    type=click.Tuple([float, float]),
    default=DEFAULTS["ASPECT"],
    help="Min,max aspect ratio (≥1, where 1 is square)",
)
# Highways
@click.option(
    "--buffer-m",
    "buffer_m",
    type=float,
    default=DEFAULTS["BUFFER_M"],
    help="Buffer around highways for context/assignment",
)
# Misc
@click.option("--skip-highways", is_flag=True, help="Skip OSM highways fetch step")
@click.option("--quiet", is_flag=True, help="Less console output")
def cli(
    aoi: str | None,
    bbox: Tuple[float, float, float, float] | None,
    out_dir: Path,
    upscale: float,
    patch: int,
    overlap: float,
    iou: float,
    conf: float,
    ndvi_min: float,
    area_m2: Tuple[float, float],
    aspect: Tuple[float, float],
    buffer_m: float,
    skip_highways: bool,
    quiet: bool,
) -> None:
    """Fetch NAIP, fetch OSM highways, run DeepForest crown detection, filter, visualize, and build risk outputs."""

    # exactly one of --aoi or --bbox
    if (aoi is None) == (bbox is None):
        raise click.UsageError("Specify exactly one of --aoi or --bbox")

    aoi_bbox = AOIS[aoi] if aoi is not None else bbox
    out_dir.mkdir(parents=True, exist_ok=True)

    _echo(f"[1/6] Fetching & clipping NAIP for AOI: {aoi_bbox} → {out_dir}", quiet)
    tif_path = fetch_and_clip_naip(aoi_bbox, out_dir)

    roads_gpkg: Path | None = None
    if not skip_highways:
        _echo(f"[2/6] Fetching OSM highways, buffer {buffer_m} m", quiet)
        roads_gpkg = out_dir / "highways_aoi.gpkg"
        fetch_highways(
            aoi_bbox=aoi_bbox,
            raster_tif=tif_path,
            out_gpkg=roads_gpkg,
            buffer_m=buffer_m,
        )

    _echo("[3/6] Running DeepForest (tiled)", quiet)
    crowns_gdf, (resx, resy), preds_df = detect_on_tif(
        tif_path,
        patch=patch,
        overlap=overlap,
        iou=iou,
        conf=conf,
        upscale=upscale,
    )

    _echo("[4/6] Applying size/shape filters and NDVI gating", quiet)
    crowns_gdf = size_shape_filter(
        crowns_gdf, preds_df, resx, resy, area_m2=area_m2, aspect=aspect
    )
    crowns_gdf = ndvi_gate(crowns_gdf, tif_path, ndvi_min=ndvi_min)

    out_crowns = out_dir / "crowns.gpkg"
    crowns_gdf.to_file(out_crowns, layer="crowns_all", driver="GPKG")

    # Risk (needs highways)
    risk_tif = risk_png = risk_spans_gpkg = risk_map_html = None
    if roads_gpkg is not None:
        _echo("[5/6] Building risk raster + risk spans", quiet)
        risk_tif, risk_png = make_risk_maps(
            tif_path,
            roads_gpkg,
            crowns_gdf,
            out_dir / "risk.tif",
            out_dir / "risk_overlay.png",
            buf_for_lines_m=1.0,
            decay_half_m=20.0,
            crown_blur_px=3,
            crown_weight=0.6,
        )
        spans_gdf, risk_spans_gpkg = risk_spans_from_raster(
            tif_path,
            roads_gpkg,
            crowns_gdf,
            out_dir / "risk_spans.gpkg",
            risk_tif=risk_tif,
            seg_len_m=50.0,
            buf_m=15.0,
        )
        
        risk_map_html = folium_risk_map(aoi_bbox, spans_gdf, out_dir / "risk_map.html")

    _echo("[6/6] Saving overlay PNG", quiet)
    out_png = out_dir / "overlay_crowns_rgb.png"
    overlay_png(tif_path, crowns_gdf, out_png)

    # Summary
    summary: dict[str, str | int] = {
        "naip_tif": str(tif_path),
        "crowns_gpkg": str(out_crowns),
        "overlay_png": str(out_png),
        "n_crowns": int(len(crowns_gdf)),
    }
    if roads_gpkg:
        summary["roads_gpkg"] = str(roads_gpkg)
    if risk_tif:
        summary["risk_tif"] = str(risk_tif)
    if risk_png:
        summary["risk_png"] = str(risk_png)
    if risk_spans_gpkg:
        summary["risk_spans_gpkg"] = str(risk_spans_gpkg)
    if risk_map_html:
        summary["risk_map_html"] = str(risk_map_html)

    if not quiet:
        click.echo(json.dumps(summary, indent=2, ensure_ascii=False))
