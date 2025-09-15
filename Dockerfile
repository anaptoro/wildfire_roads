# Dockerfile
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /code


RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential \
    gdal-bin libgdal-dev \
    libspatialindex-dev \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt constraints.txt ./


RUN python -m pip install -U pip setuptools wheel && \
    python -m pip install --no-cache-dir --no-deps supervision==0.26.1 && \
    python -m pip install --no-cache-dir -r requirements.txt -c constraints.txt && \
    python -m pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless==4.12.0.88


COPY src ./src
COPY run.py README.md ./


CMD ["python", "run.py", "--help"]
