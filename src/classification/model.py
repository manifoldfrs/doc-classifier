###############################################################################
# src/classification/model.py
# -----------------------------------------------------------------------------
# Machine-learning model wrapper (Step 4.5)
#
# This module provides a **thin** abstraction around the transformer-based document
# classifier used by the *text* and *ocr* stages. The public surface area is
# intentionally minimal:
#
# • ``predict(text: str) -> tuple[str | None, float | None]``
#     Returns a *(label, confidence)* tuple or raises when no model is loaded.
#
# Design constraints & rationale
# ==============================
# 1. **Lazy loading** – the model is loaded only on the
#    *first* call to :pyfunc:`_get_model()` to avoid incurring start-up latency
#    for requests that never hit the text/OCR stages (e.g. early filename exit).
# 2. **Thread-safety** – the loader relies on the GIL for synchronisation; no
#    explicit locks are required because the worst-case scenario is two threads
#    loading the same model concurrently which is benign.
# 3. **Strict typing** – all functions include precise type hints so `mypy
#    --strict` passes.  The implementation purposely avoids generics to keep
#    cognitive load low.
# 4. **≤ 40 lines per function** – in-line helper functions partition logic to
#    meet the repository engineering rules.
# 5. **Graceful degradation** – when the model is missing or corrupt,
#    the module raises appropriate exceptions which
#    are caught by caller stages; this behaviour keeps the pipeline functional
#    even before the model is properly configured.
###############################################################################

from __future__ import annotations

import asyncio
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, cast

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

__all__: list[str] = [
    "predict",
    "ModelNotAvailableError",
]


class ModelNotAvailableError(RuntimeError):
    """Raised when the persisted ML model artefact cannot be loaded."""


class _ModelContainer:
    """Container for the DistilBERT tokenizer and model."""

    def __init__(
        self,
        tokenizer: DistilBertTokenizer,
        model: DistilBertForSequenceClassification,
        id2label: Dict[int, str],
    ) -> None:
        self.tokenizer: DistilBertTokenizer = tokenizer
        self.model: DistilBertForSequenceClassification = model
        self.id2label: Dict[int, str] = id2label

    def predict(self, text: str) -> Tuple[str, float]:
        """Return *(label, probability)* for **text** via transformer model."""
        # Truncate text if it's too long (DistilBERT has a 512 token limit)
        text = text[:10000]  # Reasonable limit to avoid memory issues

        # Tokenize and prepare inputs
        inputs = self.tokenizer(
            text, truncation=True, padding=True, return_tensors="pt", max_length=512
        )

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Convert to probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Get the prediction
            predicted_class_id = probs.argmax().item()
            confidence = probs[0, predicted_class_id].item()

            # Get the label from the id
            predicted_label = self.id2label[predicted_class_id]

        return predicted_label, float(confidence)


# Default model paths
_DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[2] / "datasets" / "distilbert_model"
)
_DEFAULT_CONFIG_PATH = _DEFAULT_MODEL_DIR / "config.json"


def _load_distilbert(model_dir: Path) -> _ModelContainer:
    """Load the DistilBERT tokenizer, model and label mapping.

    Raises
    ------
    FileNotFoundError
        When the model directory does not exist.
    RuntimeError
        When the model cannot be loaded properly.
    """
    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(
            f"Model directory not found at '{model_dir}'. Make sure to train and save "
            "the DistilBERT model first."
        )

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Model config not found at '{config_path}'. Config file is required."
        )

    # Load id2label mapping from config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Check if id2label is in the config
    if "id2label" not in config:
        raise RuntimeError(
            "Model config is missing 'id2label' mapping. Cannot determine label names."
        )

    id2label = {int(k): v for k, v in config["id2label"].items()}

    try:
        # Load tokenizer and model from directory
        tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        model = DistilBertForSequenceClassification.from_pretrained(model_dir)

        # Set to evaluation mode
        model.eval()

    except (OSError, ValueError) as e:
        raise RuntimeError(f"Failed to load DistilBERT model: {str(e)}") from e

    return _ModelContainer(tokenizer, model, id2label)


@lru_cache(maxsize=1)
def _get_model(model_dir: Path = _DEFAULT_MODEL_DIR) -> _ModelContainer:
    """Return the singleton :class:`_ModelContainer`, loading lazily."""
    return _load_distilbert(model_dir)


def predict(text: str) -> Tuple[str | None, float | None]:
    """Predict document label for **text** using the trained DistilBERT classifier.

    Parameters
    ----------
    text:
        Pre-processed string extracted from a document.

    Returns
    -------
    tuple[str | None, float | None]
        • **label** – Highest-probability class predicted by the model.
        • **probability** – Posterior probability of *label* in the range [0, 1].

    Raises
    ------
    ModelNotAvailableError
        When the model cannot be loaded. Callers are
        expected to catch this error and apply fallback heuristics.
    """
    if not text.strip():
        return None, None

    try:
        model = _get_model()
    except (FileNotFoundError, RuntimeError) as exc:
        raise ModelNotAvailableError(str(exc)) from exc

    return model.predict(text)
