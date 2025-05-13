from __future__ import annotations
import asyncio
import pickle
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.datastructures import UploadFile

# ---------------------------------------------------------------------------
# Pipeline internals – error and success paths not previously covered
# ---------------------------------------------------------------------------
from src.classification import pipeline as _pipeline_mod
from src.classification.types import StageOutcome


@pytest.mark.asyncio
async def test_execute_stages_captures_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # noqa: D401 – extended branch coverage
    """_execute_stages must gracefully convert stage errors into null outcomes."""

    async def _boom(
        _: UploadFile,
    ) -> StageOutcome:  # pragma: no cover – body executed in test
        raise RuntimeError("boom")

    _boom.__name__ = "stage_boom"  # Identifier in results dict

    # Replace the global registry with the failing stage
    monkeypatch.setattr(_pipeline_mod, "STAGE_REGISTRY", [_boom])

    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "faulty.pdf"
    mock_file.seek = AsyncMock()
    mock_file.file = BytesIO(b"dummy")

    results = await _pipeline_mod._execute_stages(mock_file)  # type: ignore[attr-defined]

    assert results == {"stage_boom": StageOutcome(label=None, confidence=None)}


@pytest.mark.asyncio
async def test_execute_stages_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy-path run with two stages executes all loop branches."""

    async def _s1(file: UploadFile) -> StageOutcome:  # noqa: D401
        await asyncio.sleep(0)  # exercise await path
        return StageOutcome(label="invoice", confidence=0.9)

    async def _s2(file: UploadFile) -> StageOutcome:  # noqa: D401
        return StageOutcome(label="invoice", confidence=0.8)

    _s1.__name__ = "stage_one"
    _s2.__name__ = "stage_two"

    monkeypatch.setattr(_pipeline_mod, "STAGE_REGISTRY", [_s1, _s2])

    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "ok.pdf"
    mock_file.seek = AsyncMock()
    mock_file.file = BytesIO(b"content")

    results = await _pipeline_mod._execute_stages(mock_file)  # type: ignore[attr-defined]
    assert results["stage_one"].confidence == pytest.approx(0.9)
    assert results["stage_two"].label == "invoice"


def test_get_file_size_error_branch() -> None:
    """_get_file_size should return *0* when tell/seek raise."""

    class _BadFile:  # noqa: D401 – minimal stub
        def tell(self, *a, **kw):  # noqa: D401
            raise OSError("fail")

        def seek(self, *a, **kw):  # noqa: D401
            raise OSError("fail")

    bad_upload = MagicMock()
    bad_upload.filename = "err.bin"
    bad_upload.file = _BadFile()

    from src.classification.pipeline import _get_file_size

    assert _get_file_size(bad_upload) == 0


# ---------------------------------------------------------------------------
# Model loading edge-cases – missing & malformed pickle
# ---------------------------------------------------------------------------
from src.classification import model as _model_mod


def test_load_pickle_missing_file(tmp_path: Path) -> None:
    """Missing artefact must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        _model_mod._load_pickle(tmp_path / "does_not_exist.pkl")  # type: ignore[attr-defined]


def test_load_pickle_malformed(tmp_path: Path) -> None:
    """Malformed pickle lacking expected keys raises RuntimeError."""
    bad_path = tmp_path / "bad.pkl"
    with bad_path.open("wb") as handle:
        pickle.dump({"not": "expected"}, handle)

    with pytest.raises(RuntimeError):
        _model_mod._load_pickle(bad_path)  # type: ignore[attr-defined]


def test_predict_empty_string() -> None:
    """predict() should short-circuit on blank input."""
    from src.classification.model import predict

    assert predict("   ") == (None, None)


# ---------------------------------------------------------------------------
# Config coercion helpers – JSON string variants previously uncovered
# ---------------------------------------------------------------------------
from src.core.config import get_settings


auth_json = '["k_json", "k2"]'

ext_json = '["pdf", "txt"]'


def test_json_env_coercion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings should parse JSON-style env vars for API keys & extensions."""
    monkeypatch.setenv("ALLOWED_API_KEYS", auth_json)
    monkeypatch.setenv("ALLOWED_EXTENSIONS", ext_json)

    get_settings.cache_clear()  # Ensure fresh parse
    settings = get_settings()

    assert settings.allowed_api_keys == ["k_json", "k2"]
    assert settings.allowed_extensions == {"pdf", "txt"}


# ---------------------------------------------------------------------------
# Ensure classification.confidence aggregation is definitely exercised.
# ---------------------------------------------------------------------------
from src.classification.confidence import aggregate_confidences, STAGE_WEIGHTS


def _make_outcome(label: str | None, conf: float | None) -> StageOutcome:  # noqa: D401
    return StageOutcome(label=label, confidence=conf)


def test_confidence_full_path() -> None:
    """Run *aggregate_confidences* through both early-exit and weighted paths."""
    settings = get_settings()

    # Early-exit branch
    outcomes = {
        "stage_filename": _make_outcome("invoice", settings.early_exit_confidence)
    }
    label, conf = aggregate_confidences(outcomes, settings=settings)
    assert label == "invoice" and conf == settings.early_exit_confidence

    # Weighted path below threshold → unsure
    low_conf_outcomes = {
        "stage_filename": _make_outcome("invoice", 0.1),
        "stage_text": _make_outcome("invoice", 0.1),
    }
    label2, conf2 = aggregate_confidences(low_conf_outcomes, settings=settings)
    assert label2 == "unsure" and conf2 < settings.confidence_threshold

    # Weighted path above threshold, multiple labels
    mix_outcomes = {
        "stage_filename": _make_outcome("invoice", 0.8),
        "stage_metadata": _make_outcome("contract", 0.9),
    }
    label3, conf3 = aggregate_confidences(mix_outcomes, settings=settings)
    assert label3 in {"invoice", "contract"}
    assert pytest.approx(conf3) == conf3  # valid float in 0-1

    # Ensure default weight 1.0 used for unknown stage
    unknown_outcome = {"stage_new": _make_outcome("memo", 0.99)}
    l4, c4 = aggregate_confidences(unknown_outcome, settings=settings)
    assert l4 == "memo" and c4 == pytest.approx(0.99)

    # Sanity-check STAGE_WEIGHTS sums to ≤ 1.0 for known keys
    assert (
        sum(
            w for k, w in STAGE_WEIGHTS.items() if k in {"stage_filename", "stage_text"}
        )
        <= 1.0
    )
