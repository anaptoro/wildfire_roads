# Wildfire Roads – Tree Crowns & Risk (NAIP + OSM)

This repo provides an **end-to-end workflow** to:
- fetch NAIP imagery,
- fetch OpenStreetMap highways,
- detect **tree crowns** with [DeepForest],
- build a simple **risk map** around roads (e.g., for wildfire prevention / vegetation management).

> Inputs: NAIP COGs (via STAC) and OSM highways  
> Crown detector: DeepForest (PyTorch)

---

## Quick start (Docker)

**Requirements:** Docker & Docker Compose v2
**Usage**
###Build and start the image

docker build -t wildfire-pl:cpu .

###Example usage:

docker compose exec wildfire \
  python run.py --bbox -122.676 45.412 -122.666 45.422 --out outputs_small_east

## Expected outputs
- naip_tif for the selected bbox
- crowns_gpkg: Identified crowns gpkg,
- roads_gpkg: Identified roads gpkg,
- risk_tif: A tiff varying from 0-1 where risk* of wildfire is higher,
- risk_spans_gpkg: Roads under risk,
- risk_map_html: html file with the risk

## Methodology

### Crown identification
The crown identification was performed using the DeepForest model, which uses Pytorch, that was not trained on NAIP imagery, so the performance for NAIP wont be excellent, just enough for a first analysis.

There is a config file (config.py) for model calibration if needed

### Highways fetching
The fetched highways came from OpenStreet map, depending on the region many roads can be missing (a good opportunity to contribute for OSM btw).

### Risk calculation
The risk is calculated by overlapping the identified tree crowns with a buffered vector of the existing highways. After that this equation is used:

Risk per pixel is computed as a weighted blend of proximity to the nearest road and crown density—specifically 
risk=(1−crown_weight)⋅𝑃 + crown_weight⋅(𝑃⋅𝐷), where 𝑃 is an exponential distance‐decay from buffered roads (half-life = decay_half_m) and 𝐷 is a blurred, normalized raster of tree crowns (0-1).


### Use cases and caveats

This pipeline can be used for a quick overview of wildfire risk near to roads, it can be modified for powerline risk and any other feature available in OSM. 

This pipeline contain many caveats that can be addressed for more specific approaches, e.g. the fact that it currently only works with NAIP images, and the risk equation can probably have many more variables.

