"""
Decision kernel for combining stage confidence scores

This module implements the "aggregator" - the decision-making kernel of the
classification pipeline that combines the individual stage confidence scores
into a final classification with confidence.

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

from typing import Any, Dict, Tuple

from src.core.config import Settings

# Stage weights - determine how much each stage contributes to final decision
# Further increased filename weight, further decreased text/ocr weights.
STAGE_WEIGHTS: Dict[str, float] = {
    "stage_filename": 0.40,  # Increased from 0.30
    "stage_metadata": 0.20,  # Decreased from 0.25
    "stage_text": 0.20,  # Decreased from 0.25
    "stage_ocr": 0.20,  # Kept at 0.20
}


def aggregate_confidences(
    outcomes: Dict[str, Any], *, settings: Settings
) -> Tuple[str, float]:
    """
    Combine stage outcomes using weighted aggregation with optional early exit.

    This is the decision-making "kernel" of the pipeline, applying three rules:

    1. EARLY EXIT: If any stage has confidence >= early_exit_confidence,
       return its label immediately (pick highest if multiple qualify).
    2. WEIGHTED SUM: Otherwise combine per-stage scores using STAGE_WEIGHTS.
    3. THRESHOLD: If final score < confidence_threshold, return "unsure".

    Args:
        outcomes: Dictionary mapping stage names to StageOutcome objects
        settings: Application settings with threshold values

    Returns:
        Tuple of (label, confidence)
    """
    # Special case: no outcomes or no valid outcomes
    if not outcomes:
        return "unknown", 0.0

    # Check for early exit - any stage with very high confidence
    early_exit_candidates = []
    for _, outcome in outcomes.items():
        # Ensure outcome and confidence are not None before comparison
        if (
            outcome
            and outcome.label
            and outcome.confidence is not None
            and outcome.confidence >= settings.early_exit_confidence
        ):
            early_exit_candidates.append((outcome.label, outcome.confidence))

    if early_exit_candidates:
        # Take the one with highest confidence
        early_exit_candidates.sort(key=lambda x: x[1], reverse=True)
        return early_exit_candidates[0]

    # Calculate weighted scores by label
    label_scores: Dict[str, float] = {}
    label_weights: Dict[str, float] = {}

    for stage_name, outcome in outcomes.items():
        # Skip if outcome is missing, has no label, or no confidence
        if not outcome or not outcome.label or outcome.confidence is None:
            continue

        weight = STAGE_WEIGHTS.get(
            stage_name, 1.0
        )  # Default weight 1.0 for unknown stages
        weighted_score = outcome.confidence * weight

        # Initialize if label seen for the first time
        label_scores.setdefault(outcome.label, 0.0)
        label_weights.setdefault(outcome.label, 0.0)

        label_scores[outcome.label] += weighted_score
        label_weights[outcome.label] += weight

    # No valid scores found across all stages
    if not label_scores:
        return "unknown", 0.0

    # Find label with highest total weighted score
    best_label = max(label_scores.items(), key=lambda item: item[1], default=(None, 0))[
        0
    ]

    # If max returned the default (None), or if the best_label has zero total weight (e.g. all its stages had zero weight)
    if best_label is None or label_weights.get(best_label, 0.0) == 0.0:
        return "unknown", 0.0

    # Confidence is the total weighted score for that label divided by the total weight applied to that label
    confidence = label_scores[best_label] / label_weights[best_label]

    # Return "unsure" if confidence is below threshold
    if confidence < settings.confidence_threshold:
        return "unsure", confidence

    # Return the best label and its calculated confidence
    return best_label, confidence
