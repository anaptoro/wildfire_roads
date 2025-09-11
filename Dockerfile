FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NO_ALBUMENTATIONS_UPDATE=1 \
    PROJ_NETWORK=ON \
    GEOPANDAS_IO_ENGINE=pyogrio

# System geospatial stack from Debian (avoids compiling GDAL bindings)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential \
    gdal-bin libgdal-dev \
    python3-rasterio python3-fiona python3-geopandas \
    python3-pyproj python3-rtree python3-shapely python3-pyogrio \
    python3-pyparsing \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Python deps (everything except the geo stack above)
COPY requirements.txt constraints.txt ./
RUN python -m pip install -U pip setuptools wheel && \
    python -m pip install --no-cache-dir -r requirements.txt -c constraints.txt \
      --extra-index-url https://download.pytorch.org/whl/cpu

# Your app
COPY src ./src
COPY run.py README.md ./

# Idle by default; use `docker compose exec` to run jobs
CMD ["/bin/sh","-lc","sleep infinity"]
