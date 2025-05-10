#!/usr/bin/env bash
# =============================================================================
# lint.sh – Project-wide static analysis helper
# =============================================================================
# This script aggregates all mandatory code-quality checks into a single, easy
# to remember command:
#
#     ./scripts/lint.sh
#
# The same checks run in CI (see .github/workflows/ci.yml) so running this
# locally lets developers catch issues before pushing.
# =============================================================================

set -euo pipefail

# Navigate to repository root regardless of where the script is invoked.
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${PROJECT_ROOT}" ]]; then
  echo "Error: must run inside a git repository" >&2
  exit 1
fi
cd "${PROJECT_ROOT}"

# -----------------------------------------------------------------------------
# Black – code formatting (auto-format in place)
# -----------------------------------------------------------------------------
echo "▶ Running Black (auto-format)…"
python -m black src tests scripts

# -----------------------------------------------------------------------------
# isort – import sorting (auto-sort in place)
# -----------------------------------------------------------------------------
echo "▶ Running isort (auto-sort)…"
python -m isort src tests scripts

# -----------------------------------------------------------------------------
# Ruff – fast linting & import-sorting checks
# -----------------------------------------------------------------------------
echo "\n▶ Running Ruff…"
python -m ruff check src tests scripts

# -----------------------------------------------------------------------------
# mypy – static type checking (strict mode)
# -----------------------------------------------------------------------------
# We pin to Python 3.11 but allow local overrides via env var.
PYTHON_VERSION="${PY_VER:-3.11}"

echo "\n▶ Running mypy (strict)…"
python -m mypy --python-version "${PYTHON_VERSION}" src

echo "\n✅ All lint checks passed!"