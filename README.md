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
```bash
docker build -t wildfire-pl:cpu .

docker compose up -d wildfire

### 1) Run the command