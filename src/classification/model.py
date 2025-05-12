###############################################################################
# src/classification/model.py
# -----------------------------------------------------------------------------
# Machine-learning model wrapper (Step 4.5)
#
# This module provides a **thin** abstraction around the statistical document
# classifier used by the *text* and *ocr* stages.  The public surface area is
# intentionally minimal:
#
# • ``predict(text: str) -> tuple[str | None, float | None]``
#     Returns a *(label, confidence)* tuple or raises when no model is loaded.
#
# Design constraints & rationale
# ==============================
# 1. **Lazy loading** – the pickle artefact (≈ <100 KB) is loaded only on the
#    *first* call to :pyfunc:`_get_model()` to avoid incurring start-up latency
#    for requests that never hit the text/OCR stages (e.g. early filename exit).
# 2. **Thread-safety** – the loader relies on the GIL for synchronisation; no
#    explicit locks are required because the worst-case scenario is two threads
#    loading the same small pickle concurrently which is benign.
# 3. **Strict typing** – all functions include precise type hints so `mypy
#    --strict` passes.  The implementation purposely avoids generics to keep
#    cognitive load low.
# 4. **≤ 40 lines per function** – in-line helper functions partition logic to
#    meet the repository engineering rules.
# 5. **Graceful degradation** – when the model artefact is missing or corrupt,
#    the module raises :class:`FileNotFoundError` or :class:`RuntimeError` which
#    are caught by caller stages; this behaviour keeps the pipeline functional
#    even before the artefact is generated.
###############################################################################

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

__all__: list[str] = [
    "predict",
    "ModelNotAvailableError",
]


class ModelNotAvailableError(RuntimeError):
    """Raised when the persisted ML model artefact cannot be loaded."""


class _ModelContainer:  # noqa: D101 – private quasi-struct
    """Simple container for the vectoriser and estimator."""

    def __init__(self, vectoriser: TfidfVectorizer, estimator: MultinomialNB) -> None:
        self.vectoriser: TfidfVectorizer = vectoriser
        self.estimator: MultinomialNB = estimator

    def predict(self, text: str) -> Tuple[str, float]:  # noqa: D401
        """Return *(label, probability)* for **text** via NB posterior."""

        X = self.vectoriser.transform([text])
        probas = self.estimator.predict_proba(X)[0]
        predicted_index: int = probas.argmax()
        return self.estimator.classes_[predicted_index], float(probas[predicted_index])


# The model is expected at ``<repo-root>/datasets/model.pkl``.
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "datasets" / "model.pkl"


def _load_pickle(path: Path) -> _ModelContainer:  # noqa: D401 – helper
    """Load the **vectoriser** and **estimator** from *path*.

    Raises
    ------
    FileNotFoundError
        When the artefact does not exist.
    RuntimeError
        When the pickle does not contain the expected keys.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Model artefact not found at '{path}'. Run 'scripts/train_model.py' "
            "to generate it."
        )

    with path.open("rb") as handle:
        data: Any = pickle.load(handle)

    if (
        not isinstance(data, dict)
        or {
            "vectoriser",
            "estimator",
        }
        - data.keys()
    ):  # noqa: E713  – explicit diff check
        raise RuntimeError(
            "model.pkl is malformed – expected a dict with keys 'vectoriser' and 'estimator'."
        )

    vectoriser = data["vectoriser"]
    estimator = data["estimator"]
    if not isinstance(vectoriser, TfidfVectorizer) or not isinstance(
        estimator, MultinomialNB
    ):
        raise RuntimeError("model.pkl contains objects of unexpected types.")

    return _ModelContainer(vectoriser, estimator)


@lru_cache(maxsize=1)
def _get_model(path: Path = _DEFAULT_MODEL_PATH) -> _ModelContainer:  # noqa: D401
    """Return the singleton :class:`_ModelContainer`, loading lazily."""

    return _load_pickle(path)


def predict(text: str) -> Tuple[str | None, float | None]:  # noqa: D401
    """Predict document label for **text** using the trained NB classifier.

    Parameters
    ----------
    text:
        Pre-processed string (typically lower-cased) extracted from a document.

    Returns
    -------
    tuple[str | None, float | None]
        • **label** – Highest-probability class predicted by the model.
        • **probability** – Posterior probability of *label* in the range [0, 1].

    Raises
    ------
    ModelNotAvailableError
        When the persisted model artefact cannot be loaded.  Callers are
        expected to catch this error and apply fallback heuristics.
    """

    if not text.strip():
        return None, None

    try:
        model = _get_model()
    except (FileNotFoundError, RuntimeError) as exc:  # pragma: no cover
        raise ModelNotAvailableError(str(exc)) from exc

    return model.predict(text)
