from unittest.mock import AsyncMock, MagicMock

import pytest

from src.classification.stages import text as _text_mod
from src.classification.stages.text import stage_text


@pytest.mark.asyncio
async def test_stage_text_heuristic_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When model unavailable, stage should fall back to heuristics and match invoice pattern."""

    # Disable ML model path
    monkeypatch.setattr(_text_mod, "_MODEL_AVAILABLE", False)

    # Create mock UploadFile for a .txt file
    mock_file = MagicMock()
    mock_file.filename = "payment_invoice.txt"
    mock_file.seek = AsyncMock()
    mock_file.read = AsyncMock(
        return_value=b"Total amount due: 100. Please pay this invoice immediately."
    )

    outcome = await stage_text(mock_file)
    assert outcome.label == "invoice" and outcome.confidence == pytest.approx(0.75)
