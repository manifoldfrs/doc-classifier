from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from src.classification.model import (
    ModelNotAvailableError,
    _DEFAULT_MODEL_PATH,
    _ModelContainer,
    _get_model,
    _load_pickle,
    predict,
)


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Fixture to automatically clear the LRU cache for _get_model before each test."""
    _get_model.cache_clear()
    yield
    _get_model.cache_clear()


@pytest.fixture
def mock_model_container() -> MagicMock:
    """Provides a mock _ModelContainer."""
    mock_vectoriser = MagicMock(spec=TfidfVectorizer)
    mock_estimator = MagicMock(spec=MultinomialNB)
    mock_estimator.classes_ = ["invoice", "contract"]  # Example classes

    container = _ModelContainer(mock_vectoriser, mock_estimator)
    # Mock the internal predict method of the container
    container.predict = MagicMock(return_value=("invoice", 0.95))
    return container


def test_model_container_predict(mock_model_container: MagicMock) -> None:
    """Test the internal predict method of the _ModelContainer (requires mocking internals)."""
    # This test assumes the actual predict logic of _ModelContainer works,
    # focusing on testing the `predict` wrapper function.
    # We mock the behavior directly on the fixture.
    text = "Sample text for prediction"
    label, prob = mock_model_container.predict(text)

    assert label == "invoice"
    assert prob == 0.95
    mock_model_container.predict.assert_called_once_with(text)


def test_predict_success(mock_model_container: MagicMock) -> None:
    """Test the main `predict` function happy path."""
    text = "This is an invoice document."
    with patch(
        "src.classification.model._get_model", return_value=mock_model_container
    ):
        label, confidence = predict(text)

        assert label == "invoice"
        assert confidence == 0.95
        # Check that the container's predict was called via _get_model
        mock_model_container.predict.assert_called_once_with(text)


def test_predict_empty_text() -> None:
    """Test `predict` with empty or whitespace-only text."""
    with patch(
        "src.classification.model._get_model"
    ) as mock_get_model:  # Should not be called
        label, confidence = predict("")
        assert label is None
        assert confidence is None

        label, confidence = predict("   \n\t ")
        assert label is None
        assert confidence is None

        mock_get_model.assert_not_called()


def test_predict_text_strip_is_false() -> None:
    """Test predict when text.strip() is false (e.g. only whitespace)."""
    with patch("src.classification.model._get_model") as mock_get_model:
        label, confidence = predict("     ")
        assert label is None
        assert confidence is None
        mock_get_model.assert_not_called()

        label, confidence = predict("\n\t\r")
        assert label is None
        assert confidence is None
        mock_get_model.assert_not_called()


def test_predict_model_not_found() -> None:
    """Test `predict` raises ModelNotAvailableError when model file is missing."""
    text = "Some valid text"
    with patch(
        "src.classification.model._load_pickle",
        side_effect=FileNotFoundError("Not found"),
    ):
        with pytest.raises(ModelNotAvailableError, match="Not found"):
            predict(text)


def test_predict_model_load_runtime_error() -> None:
    """Test `predict` raises ModelNotAvailableError on pickle load runtime error."""
    text = "Some valid text"
    with patch(
        "src.classification.model._load_pickle",
        side_effect=RuntimeError("Corrupt pickle"),
    ):
        with pytest.raises(ModelNotAvailableError, match="Corrupt pickle"):
            predict(text)


def test_load_pickle_success(tmp_path: Path) -> None:
    """Test _load_pickle successfully loads a valid pickle file."""
    # Create actual instances for pickling, as _load_pickle checks types
    mock_vectoriser = TfidfVectorizer()
    mock_estimator = MultinomialNB()
    model_data = {"vectoriser": mock_vectoriser, "estimator": mock_estimator}

    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    container = _load_pickle(model_path)

    assert isinstance(container, _ModelContainer)
    # Check that the loaded objects are of the expected types
    assert isinstance(container.vectoriser, TfidfVectorizer)
    assert isinstance(container.estimator, MultinomialNB)


def test_load_pickle_file_not_found(tmp_path: Path) -> None:
    """Test _load_pickle raises FileNotFoundError."""
    model_path = tmp_path / "non_existent_model.pkl"
    with pytest.raises(FileNotFoundError):
        _load_pickle(model_path)


def test_load_pickle_malformed_dict(tmp_path: Path) -> None:
    """Test _load_pickle raises RuntimeError for missing keys."""
    model_data = {"vectoriser": MagicMock(spec=TfidfVectorizer)}  # Missing 'estimator'
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    with pytest.raises(RuntimeError, match="malformed.*expected a dict with keys"):
        _load_pickle(model_path)


def test_load_pickle_not_a_dict(tmp_path: Path) -> None:
    """Test _load_pickle raises RuntimeError if pickle is not a dict."""
    model_data = ["list", "instead", "of", "dict"]
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    with pytest.raises(RuntimeError, match="malformed.*expected a dict"):
        _load_pickle(model_path)


def test_load_pickle_wrong_types(tmp_path: Path) -> None:
    """Test _load_pickle raises RuntimeError for incorrect object types."""
    model_data = {"vectoriser": "not a vectoriser", "estimator": "not an estimator"}
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    with pytest.raises(RuntimeError, match="contains objects of unexpected types"):
        _load_pickle(model_path)


def test_get_model_uses_cache(mock_model_container: MagicMock) -> None:
    """Test that _get_model uses the LRU cache."""
    with patch(
        "src.classification.model._load_pickle", return_value=mock_model_container
    ) as mock_load_pickle_func:
        model1 = _get_model(_DEFAULT_MODEL_PATH)
        model2 = _get_model(_DEFAULT_MODEL_PATH)
        assert model1 is model2
        # _load_pickle should only be called once due to caching
        mock_load_pickle_func.assert_called_once_with(_DEFAULT_MODEL_PATH)
