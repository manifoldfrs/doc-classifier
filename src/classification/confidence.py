"""src/classification/confidence.py
###############################################################################
Confidence aggregation utilities (Step 4.4)
###############################################################################
This module consolidates **per-stage** confidence scores produced by the
multi-stage document-classification pipeline into a single _final_ score &
label.  It implements the rules laid out in the technical specification:

1. **Early-exit optimisation** – if any stage reports a confidence greater than
   or equal to the configurable ``EARLY_EXIT_CONFIDENCE`` (default = 0.9) the
   pipeline short-circuits and returns that stage ⇢ label  ← score.
2. **Weighted aggregation** – otherwise, the module combines stage scores via a
   weighted average.  Weights are encoded in :pydata:`STAGE_WEIGHTS` and can be
   tweaked centrally without touching the pipeline orchestrator.
3. **Thresholding** – when the aggregated score falls below
   ``CONFIDENCE_THRESHOLD`` (default = 0.65) the label is downgraded to
   ``"unsure"`` so clients can apply fallback handling.

Design constraints
==================
• The public helpers are _pure_ functions – no side-effects – which makes them
  trivial to unit-test (see *tests/unit/classification/test_confidence.py* in a
  later implementation step).
• Functions respect the repository limit of **≤ 40 lines** each.
• Explicit type hints and Pydantic-based data-models guarantee mypy compliance.

Public API
==========
``aggregate_confidences()`` – main entry-point used by the pipeline.

Edge cases & validation
-----------------------
• When **no** stage yields a label the aggregator returns ``("unknown", 0.0)``.
• Unknown stage names default to a weight of 1.0 ensuring forward compatibility
  with future custom stages.

"""

from __future__ import annotations

# stdlib
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Mapping, Tuple

# local imports
from src.core.config import Settings, get_settings

__all__: list[str] = [
    "aggregate_confidences",
    "STAGE_WEIGHTS",
]

# ---------------------------------------------------------------------------
# Weight configuration – stage function __name__  → weight (float 0-1)
# The mapping lives _once_ here so changes propagate automatically.
# ---------------------------------------------------------------------------
STAGE_WEIGHTS: Dict[str, float] = {
    "stage_filename": 0.15,
    "stage_metadata": 0.25,
    "stage_text": 0.35,
    "stage_ocr": 0.25,
}

if TYPE_CHECKING:  # pragma: no cover – import solely for static analysis
    from src.classification.pipeline import StageOutcome


def aggregate_confidences(
    stage_outcomes: Mapping[str, "StageOutcome"],
    *,
    settings: Settings | None = None,
) -> Tuple[str, float]:  # noqa: D401 – helper function signature
    """Return *(label, confidence)* aggregated across stages.

    Parameters
    ----------
    stage_outcomes:
        Mapping from **stage function name** → :class:`StageOutcome` produced by
        the classification pipeline.
    settings:
        Optional :class:`~src.core.config.Settings` instance.  The cached global
        singleton is used when omitted.  Providing an explicit instance is
        useful for unit-tests that need deterministic behaviour regardless of
        ``os.environ``.
    """

    settings = settings or get_settings()

    # ------------------------------------------------------------------
    # 1. Early-exit – highest confidence ≥ early_exit_confidence
    # ------------------------------------------------------------------
    for outcome in stage_outcomes.values():
        if (
            outcome.label
            and outcome.confidence is not None
            and outcome.confidence >= settings.early_exit_confidence
        ):
            return outcome.label, float(outcome.confidence)

    # ------------------------------------------------------------------
    # 2. Weighted aggregation – accumulate per-label weighted scores
    # ------------------------------------------------------------------
    scores: Dict[str, float] = defaultdict(float)
    weights_seen: Dict[str, float] = defaultdict(float)

    for stage_name, outcome in stage_outcomes.items():
        if outcome.label and outcome.confidence is not None:
            weight: float = STAGE_WEIGHTS.get(stage_name, 1.0)
            scores[outcome.label] += outcome.confidence * weight
            weights_seen[outcome.label] += weight

    if not scores:
        return "unknown", 0.0

    # Normalise to obtain weighted averages and pick label with highest score
    aggregated_label: str = max(scores, key=scores.get)
    aggregated_confidence: float = (
        scores[aggregated_label] / weights_seen[aggregated_label]
    )

    # ------------------------------------------------------------------
    # 3. Thresholding – downgrade to "unsure" when below configured minimum
    # ------------------------------------------------------------------
    if aggregated_confidence < settings.confidence_threshold:
        return "unsure", aggregated_confidence

    return aggregated_label, aggregated_confidence
