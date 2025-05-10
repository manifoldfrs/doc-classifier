"""tests/unit/classification/test_confidence.py
###############################################################################
Unit tests for ``src.classification.confidence.aggregate_confidences``
(Implementation Plan – Step 9.2)
###############################################################################
The aggregator is the *decision kernel* of the pipeline, merging per-stage
scores into a final label + confidence.  The tests verify compliance with the
three specification clauses:

1. **Early-exit** – any stage reporting ≥ ``EARLY_EXIT_CONFIDENCE`` must short
   circuit and return its label/score.
2. **Weighted average** – absent early-exit, the function combines confidences
   proportionally to :data:`src.classification.confidence.STAGE_WEIGHTS`.
3. **Thresholding** – aggregated scores below ``CONFIDENCE_THRESHOLD`` are
   downgraded to ``"unsure"``.

We construct :class:`src.classification.pipeline.StageOutcome` instances
manually so the tests remain isolated from the full pipeline.
"""

from __future__ import annotations

# third-party
import pytest

# local
from src.classification.confidence import (
    STAGE_WEIGHTS,
    aggregate_confidences,
)
from src.classification.pipeline import StageOutcome
from src.core.config import Settings

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _out(label: str, conf: float) -> StageOutcome:  # noqa: D401 terse factory
    return StageOutcome(label=label, confidence=conf)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_early_exit_short_circuits() -> None:
    """Stage with ≥ EARLY_EXIT_CONFIDENCE should dominate regardless of others."""

    settings = Settings(early_exit_confidence=0.9)

    outcomes = {
        "stage_filename": _out("invoice", 0.95),  # high – should win
        "stage_text": _out("bank_statement", 0.8),
    }

    label, conf = aggregate_confidences(outcomes, settings=settings)

    assert label == "invoice"
    assert conf == pytest.approx(0.95)


def test_weighted_aggregation_majority_label() -> None:
    """Weighted average picks label with highest *weighted* score, not max raw."""

    settings = Settings(confidence_threshold=0.1, early_exit_confidence=0.9)

    # Two stages agree on *bank_statement* with decent scores; one minor *invoice*
    outcomes = {
        "stage_filename": _out("bank_statement", 0.8),
        "stage_metadata": _out("bank_statement", 0.6),
        "stage_text": _out("invoice", 0.7),
    }

    # Manual expected weighted calculation
    bs_score = (
        0.8 * STAGE_WEIGHTS["stage_filename"] + 0.6 * STAGE_WEIGHTS["stage_metadata"]
    )
    invoice_score = 0.7 * STAGE_WEIGHTS["stage_text"]
    expected_label = "bank_statement" if bs_score > invoice_score else "invoice"

    label, conf = aggregate_confidences(outcomes, settings=settings)

    assert label == expected_label
    # Ensure confidence reflects correct weighted average for winning label
    if label == "bank_statement":
        expected_confidence = bs_score / (
            STAGE_WEIGHTS["stage_filename"] + STAGE_WEIGHTS["stage_metadata"]
        )
    else:
        expected_confidence = invoice_score / STAGE_WEIGHTS["stage_text"]

    assert conf == pytest.approx(expected_confidence, rel=1e-5)


def test_threshold_downgrades_to_unsure() -> None:
    """Aggregated score below threshold must yield label 'unsure'."""

    settings = Settings(confidence_threshold=0.8, early_exit_confidence=0.9)

    outcomes = {
        "stage_filename": _out("invoice", 0.6),
        "stage_text": _out("invoice", 0.5),
    }

    label, conf = aggregate_confidences(outcomes, settings=settings)

    assert label == "unsure"
    assert conf < settings.confidence_threshold
