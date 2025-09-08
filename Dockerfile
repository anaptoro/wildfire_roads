FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PROJ_NETWORK=ON \
    GEOPANDAS_IO_ENGINE=pyogrio \
    PIP_CONSTRAINT=/app/constraints.txt \
    # prefer wheels where possible, but we will build Fiona/GDAL if needed
    PIP_PREFER_BINARY=1

# System GDAL + deps (versions from Debian; will define gdal-config)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      gdal-bin libgdal-dev \
      libproj-dev proj-bin \
      libgeos-dev \
      ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# These help pip find headers if it needs to build against system GDAL
ENV GDAL_CONFIG=/usr/bin/gdal-config \
    CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app
COPY constraints.txt constraints.txt
COPY requirements.txt requirements.txt

# Tooling
RUN python -m pip install --upgrade pip setuptools wheel

# 1) Install Python bindings for the *exact* system GDAL version
#    (queries gdal-config to avoid mismatches)
RUN GDAL_VER=$(gdal-config --version) && \
    echo "Installing GDAL Python bindings ${GDAL_VER}" && \
    python -m pip install "GDAL==${GDAL_VER}"

# 2) If you need Fiona, install it *after* GDAL so it builds against system GDAL
#    (Otherwise, prefer pyogrio and skip Fiona entirely.)
# RUN python -m pip install fiona

# 3) Install the rest (torch CPU wheels if used)
RUN python -m pip install -r requirements.txt -c constraints.txt \
      --extra-index-url https://download.pytorch.org/whl/cpu

# App code
COPY src ./src
COPY run.py ./run.py
COPY README.md ./README.md

# Quick sanity
RUN python - <<'PY'
from osgeo import gdal
import geopandas as gpd
print("GDAL:", gdal.VersionInfo(), "GeoPandas:", gpd.__version__)
PY

ENTRYPOINT ["python", "run.py"]
