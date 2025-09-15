# Dockerfile
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /code

# OS deps (GDAL runtime, RTree, and a couple of harmless GLib libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential \
    gdal-bin libgdal-dev \
    libspatialindex-dev \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy lock files first for better layer caching
COPY requirements.txt constraints.txt ./

# Install: prevent supervision from dragging in opencv-python,
# then install your requirements, and finally force headless cv2.
RUN python -m pip install -U pip setuptools wheel && \
    python -m pip install --no-cache-dir --no-deps supervision==0.26.1 && \
    python -m pip install --no-cache-dir -r requirements.txt -c constraints.txt && \
    python -m pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless==4.12.0.88

# App code
COPY src ./src
COPY run.py README.md ./

# Default command (optional)
CMD ["python", "run.py", "--help"]
