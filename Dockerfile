###############################################################################
# Dockerfile ─ HeronAI Document-Classifier Demo
# -----------------------------------------------------------------------------
# Multi-stage build that produces a **slim, production-oriented** container
# image while keeping build-time artefacts out of the final layer.
#
#   Stage 0  →  "builder"     • installs Python deps into /install
#   Stage 1  →  "runtime"     • copies deps + source, adds system libs, runs app
#
# Key design decisions
# ====================
# • Base image:  python:3.11-slim  → smallest official image that still has
#   GLibc (required by many-linux wheels) and apt package manager.
# • System libs: Tesseract OCR & friends are installed only in the final stage
#   to keep security-scanning focused on what actually ships.
# • Non-root user could be added, but Vercel containers already run as non-root.
# • PORT env default = 8000 so the same container works in Docker Compose,
#   Kubernetes, and Vercel's container runtime.
#
# Build:
#   docker build -t heronai:latest .
#
# Run (stand-alone):
#   docker run -p 8000:8000 heronai:latest
###############################################################################

############################
# Stage 0 – dependency build
############################
FROM python:3.11-slim AS builder

# Basic hygiene
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /src

# System toolchain for wheels requiring native extensions (pandas, Pillow)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into an isolated prefix we can later copy
COPY requirements.txt .
RUN pip install --upgrade --no-cache-dir pip setuptools wheel \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

############################
# Stage 1 – slim runtime
############################
FROM python:3.11-slim

LABEL org.opencontainers.image.title="HeronAI Document Classifier Demo" \
      org.opencontainers.image.description="FastAPI micro-service for high-throughput document classification." \
      org.opencontainers.image.source="https://github.com/<your-org>/heronai-doc-classifier"

# Runtime env vars
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# System packages required **at runtime**
# • tesseract-ocr + libtesseract-dev → OCR stage
# • libgl1 + libglib2.0-0           → Pillow/OpenCV backend symbols
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr libtesseract-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels/site-packages produced in *builder* stage
COPY --from=builder /install /usr/local

# Copy application source
WORKDIR /app
COPY . .

# Network port
EXPOSE 8000

# Health-aware startup could be added with /usr/bin/dumb-init – out of scope now
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
