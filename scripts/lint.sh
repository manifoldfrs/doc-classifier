#!/usr/bin/env bash
set -euo pipefail

# Navigate to repository root regardless of where the script is invoked.
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${PROJECT_ROOT}" ]]; then
  echo "Error: must run inside a git repository" >&2
  exit 1
fi
cd "${PROJECT_ROOT}"

###############################################################################
# 1. Code formatting – Black
###############################################################################
echo "▶ Running Black (auto-format)…"
python -m black src tests scripts

###############################################################################
# 2. Import sorting & lint auto-fix – Ruff
###############################################################################
# Ruff can both *organise imports* and fix safe violations.  We first run it in
# *--fix* mode to mutate the working tree, then perform a second *check* pass
# to fail fast on any remaining offences that require manual attention.

echo "▶ Running Ruff (auto-fix)…"
python -m ruff check --fix src scripts

echo "▶ Running Ruff (verify)…"
python -m ruff check src scripts

# -----------------------------------------------------------------------------
# mypy – static type checking (strict mode)
# -----------------------------------------------------------------------------
# We pin to Python 3.11 but allow local overrides via env var.
PYTHON_VERSION="${PY_VER:-3.11}"

echo "\n▶ Running mypy (strict)…"
python -m mypy --python-version "${PYTHON_VERSION}" src

echo "\n✅ All lint checks passed!"