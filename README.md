# HeronAI – Document Classifier Demo

A production-leaning micro-service that classifies documents at scale.

## Prerequisites

- Python **3.11** or newer (use `pyenv` or your OS package manager)
- Git

> All commands assume a Unix-like shell (macOS / Linux) and your working
> directory is the repository root.

## Quick-start (local development)

```bash
# 1. Clone
 git clone https://github.com/<your-org>/heronai-doc-classifier.git
 cd heronai-doc-classifier

# 2. Create & activate a virtual-env
 python -m venv .venv
 source .venv/bin/activate

# 3. Install runtime + tooling dependencies
 pip install -r requirements.txt

# 4. (Optional) copy environment template and tweak
 cp .env.example .env  # if you create one – otherwise defaults apply

# 5. Run the FastAPI layer with auto-reload
 uvicorn src.api:app --reload
```

## Tests & linters

```bash
# Run static analysis & unit tests (≥ 95 % coverage expected)
ruff . && black --check . && mypy --strict src && pytest -q
```

## Common environment variables

| Variable               | Default | Description                                   |
| ---------------------- | ------- | --------------------------------------------- |
| `DEBUG`                | `false` | Enable verbose logging & hot-reload           |
| `ALLOWED_API_KEYS`     | ``      | Comma-separated static API keys               |
| `CONFIDENCE_THRESHOLD` | `0.65`  | Minimum aggregated confidence to return label |

If omitted, sane defaults defined in `src/core/config.py` are used.

## Hotkeys

- **J** – Request deeper dive into the classification pipeline
- **K** – Ask for Terraform infra spec
- **L** – Open discussion on UI scope
