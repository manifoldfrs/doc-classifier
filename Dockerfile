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

FROM python:3.11-slim

LABEL org.opencontainers.image.title="Document Classifier Demo" \
      org.opencontainers.image.description="FastAPI micro-service for high-throughput document classification." \
      org.opencontainers.image.source="https://github.com/test/document-classifier"

# Runtime env vars
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# System packages required **at runtime**
# • tesseract-ocr                     → OCR stage
# • libgl1 + libglib2.0-0           → Pillow/OpenCV backend symbols
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels/site-packages produced in *builder* stage
COPY --from=builder /install /usr/local

# Copy application source and necessary data
WORKDIR /app
COPY src ./src
COPY datasets ./datasets

# Network port
EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "$PORT"]
