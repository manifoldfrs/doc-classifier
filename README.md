# HeronAI Document Classifier

[![CI](https://github.com/yourusername/heronai-doc-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/heronai-doc-classifier/actions/workflows/ci.yml)
[![Deploy](https://github.com/yourusername/heronai-doc-classifier/actions/workflows/deploy.yml/badge.svg)](https://github.com/yourusername/heronai-doc-classifier/actions/workflows/deploy.yml)

A production-leaning document classification micro-service built with FastAPI, capable of scaling to millions of files per day while remaining simple, testable, and maintainable.

## 📚 Overview

HeronAI is a document classification service that intelligently categorizes diverse financial documents. It processes uploads through a multi-stage pipeline:

1. **Filename Analysis** → Quick heuristics based on filenames
2. **Metadata Extraction** → PDF metadata/EXIF analysis
3. **Text Content Analysis** → TF-IDF vectorization with Naive Bayes classification
4. **OCR Fallback** → Image-based text extraction when needed

The service processes files individually or in batches, providing confidence scores and detailed insights about the classification process.

## ✨ Features

- **Rich Format Support**: PDF, DOCX, CSV, images (JPG/PNG), TXT, and more
- **Multi-stage Pipeline**: Intelligent classification with early-exit optimization
- **Batch Processing**: Process up to 50 files in one request
- **Async Job Support**: Background processing for larger batches
- **REST API**: Clean, versioned endpoints with OpenAPI documentation
- **Observable**: Structured logging and Prometheus metrics
- **Containerized**: Multi-stage Docker build for deployment flexibility

## ⚡ Quick Start (Docker Compose)

```bash
# 1. Clone & enter the repo
git clone https://github.com/yourusername/heronai-doc-classifier.git
cd heronai-doc-classifier

# 2. Spin-up the full stack (FastAPI + Redis + Postgres)
docker compose up --build

# 3. Explore the live API docs
open http://localhost:8000/docs  # or simply paste into your browser
```

The container looks for a local `.env` file. If one doesn't exist yet, create it and at minimum set:

```ini
ALLOWED_API_KEYS=<your_api_key>
DEBUG=true  # optional – enables hot-reload & verbose logs
```

Generate a random API key with the helper script:

```bash
python scripts/generate_api_key.py --count 1
```

Copy the value into your `.env` (or export it as an environment variable) and include it in the `x-api-key` request header when calling the API.

## 🛠️ Prerequisites

- Python 3.11+
- Git
- Docker & Docker Compose (optional, for containerized setup)
- Tesseract OCR – required for the OCR fallback stage

## 🚀 Getting Started

### Local Development Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/heronai-doc-classifier.git
cd heronai-doc-classifier

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install pre-commit hooks
pre-commit install

# 5. Create a local environment file (if you haven't already)
cp .env.example .env  # or just touch .env and add the keys you need
# At minimum set ALLOWED_API_KEYS and CONFIDENCE_THRESHOLD if you want to tune the model.

# 🔑 Generate a random API key (optional convenience helper)
python scripts/generate_api_key.py --count 1 >> .env  # appends ALLOWED_API_KEYS=<key>

# 6. Run the application with hot-reload
uvicorn src.api.app:app --reload
```

### Docker Setup

```bash
# Build and start the complete stack (App + Redis + Postgres)
docker compose up --build

# Alternatively, just build the application image
docker build -t heronai:latest .
docker run -p 8000:8000 heronai:latest
```

## 📖 Usage

### API Endpoints

- `POST /v1/files`: Upload and classify one or more files
- `GET /v1/jobs/{job_id}`: Retrieve results for asynchronous batch jobs
- `GET /v1/health`: Health check endpoint
- `GET /v1/version`: Version information
- `GET /metrics`: Prometheus metrics (when enabled)
- `GET /docs`: OpenAPI documentation

### Example: Classifying Files

```bash
# Classify a single file
curl -X POST \
  "http://localhost:8000/v1/files" \
  -H "x-api-key: your_api_key_here" \
  -F "files=@path/to/your/document.pdf"

# Classify multiple files
curl -X POST \
  "http://localhost:8000/v1/files" \
  -H "x-api-key: your_api_key_here" \
  -F "files=@path/to/file1.pdf" \
  -F "files=@path/to/file2.jpg" \
  -F "files=@path/to/file3.docx"
```

### Example Response

```json
[
  {
    "filename": "invoice_may_2023.pdf",
    "mime_type": "application/pdf",
    "size_bytes": 84213,
    "label": "invoice",
    "confidence": 0.92,
    "stage_confidences": {
      "stage_filename": 0.8,
      "stage_metadata": 0.86,
      "stage_text": 0.92,
      "stage_ocr": null
    },
    "pipeline_version": "v1.0.0",
    "processing_ms": 137,
    "request_id": "218c2c4d-d8a4-4a27-a6b8-d20fb0afa7bd",
    "warnings": [],
    "errors": []
  }
]
```

## 🏗️ Architecture

```
                                                   ┌─────────────────┐
                                                   │  FastAPI Layer  │
                                                   │  - API Routes   │
                                                   │  - Validation   │
                                                   │  - Auth (API Key)│
                                                   └───────┬─────────┘
                                                           │
                                                           ▼
┌─────────────────┐         ┌─────────────────┐    ┌──────────────────┐
│     Parsing     │◄────────│  Classification  │◄───│    Ingestion     │
│  - PDF (pdfminer)│         │    Pipeline     │    │   - Validation   │
│  - DOCX (docx2txt)│        │  - Filename     │    │   - Streaming    │
│  - CSV (pandas)  │────────►│  - Metadata     │    │   - MIME Check   │
│  - OCR (tesseract)│        │  - Text         │    └──────────────────┘
└─────────────────┘         │  - OCR          │
                            └────────┬────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
          ┌───────────────────┐            ┌──────────────────┐
          │ Sync Response     │            │ Async Jobs       │
          │ (Small Batches)   │            │ (Large Batches)  │
          └───────────────────┘            └──────────────────┘
```

## 📊 Environment Variables

| Variable                | Default                | Description                                |
| ----------------------- | ---------------------- | ------------------------------------------ |
| `DEBUG`                 | `false`                | Enable verbose logging & hot-reload        |
| `ALLOWED_API_KEYS`      | `""`                   | Comma-separated static API keys            |
| `ALLOWED_EXTENSIONS`    | `pdf,docx,jpg,png,...` | Comma-separated allowed file extensions    |
| `MAX_FILE_SIZE_MB`      | `10`                   | Maximum file size in MB                    |
| `MAX_BATCH_SIZE`        | `50`                   | Maximum number of files per batch request  |
| `CONFIDENCE_THRESHOLD`  | `0.65`                 | Minimum confidence score to assign a label |
| `EARLY_EXIT_CONFIDENCE` | `0.9`                  | Score threshold for pipeline early-exit    |
| `PROMETHEUS_ENABLED`    | `true`                 | Toggle Prometheus metrics endpoint         |
| `PIPELINE_VERSION`      | `v0.1.0`               | Semantic version embedded in API responses |

## 🧪 Testing

```bash
# Run linting, type checking, and tests (with coverage)
./scripts/lint.sh && mypy --strict src && pytest

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run tests with coverage report
pytest --cov=src
```

## 🔍 Hotkeys

This project supports the following hotkeys for exploring specific aspects:

- **J** – Request deeper dive into the classification pipeline implementation
- **K** – Ask for Terraform infrastructure specification
- **L** – Open discussion on UI scope and frontend implementation

## 📁 Project Structure

```
src/
  api/            # FastAPI app & routers
  ingestion/      # file validation & streaming
  parsing/        # pdf.py, docx.py, csv.py, image.py
  classification/ # pipeline, stages, model, confidence
  core/           # config, logging, exceptions
  utils/          # auth, timers, helpers
scripts/          # train_model.py, generate_synthetic.py
tests/            # unit/, integration/, data/
```

## 🔒 Security & Auth

API authentication is handled via the `x-api-key` header, validated against the `ALLOWED_API_KEYS` environment variable. When this variable is empty, authentication is disabled (useful for development).

## 📚 Additional Documentation

For more details, see:

- [Limitations & Future Work](docs/limitations.md)
- [Frontend Setup Guide](docs/frontend_setup.md)

## 📄 License

This project is licensed under the [MIT License](LICENSE).
