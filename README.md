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

### 1) Build and start the image

docker build -t wildfire-pl:cpu .

###Example usage:

docker compose exec wildfire \
  python run.py --bbox -122.676 45.412 -122.666 45.422 --out outputs_small_east