from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from src.classification.model import (
    ModelNotAvailableError,
    _get_model,
    _load_pickle,
    _ModelContainer,
    predict,
)


@pytest.fixture
def mock_model_data() -> dict:
    """Provides mock vectorizer and estimator objects."""
    mock_vectorizer = MagicMock(spec=TfidfVectorizer)
    mock_estimator = MagicMock(spec=MultinomialNB)
    # Configure mock estimator's predict_proba and classes_
    mock_estimator.predict_proba.return_value = [[0.1, 0.9]]  # Example probabilities
    mock_estimator.classes_ = ["class_a", "class_b"]  # Example class labels
    # Mock vectorizer's transform
    mock_vectorizer.transform.return_value = (
        MagicMock()
    )  # Return a dummy sparse matrix or array
    return {"vectoriser": mock_vectorizer, "estimator": mock_estimator}


@pytest.fixture
def temp_model_path(tmp_path: Path, mock_model_data: dict) -> Path:
    """Creates a temporary model pickle file for testing."""
    model_file = tmp_path / "test_model.pkl"
    with model_file.open("wb") as f:
        pickle.dump(mock_model_data, f)
    return model_file


@pytest.fixture(autouse=True)
def clear_lru_cache():
    """Clears the LRU cache of _get_model before each test."""
    _get_model.cache_clear()
    yield  # Run the test
    _get_model.cache_clear()  # Clear cache after test


def test_model_container_predict(mock_model_data: dict) -> None:
    """Tests the predict method of the internal _ModelContainer."""
    container = _ModelContainer(**mock_model_data)
    text = "some input text"
    label, probability = container.predict(text)

    mock_model_data["vectoriser"].transform.assert_called_once_with([text])
    mock_model_data["estimator"].predict_proba.assert_called_once()
    # Based on mock_estimator setup: class_b has 0.9 probability
    assert label == "class_b"
    assert probability == pytest.approx(0.9)


def test_load_pickle_success(temp_model_path: Path) -> None:
    """Tests successful loading of a valid model pickle file."""
    container = _load_pickle(temp_model_path)
    assert isinstance(container, _ModelContainer)
    assert isinstance(container.vectoriser, TfidfVectorizer)
    assert isinstance(container.estimator, MultinomialNB)


def test_load_pickle_file_not_found(tmp_path: Path) -> None:
    """Tests _load_pickle when the model file does not exist."""
    non_existent_path = tmp_path / "not_a_model.pkl"
    with pytest.raises(FileNotFoundError):
        _load_pickle(non_existent_path)


def test_load_pickle_malformed_dict(tmp_path: Path) -> None:
    """Tests _load_pickle with a pickle file containing an invalid dictionary."""
    malformed_file = tmp_path / "malformed.pkl"
    with malformed_file.open("wb") as f:
        pickle.dump({"wrong_key": 123}, f)  # Missing 'vectoriser'/'estimator'

    with pytest.raises(RuntimeError, match="model.pkl is malformed"):
        _load_pickle(malformed_file)


def test_load_pickle_wrong_types(tmp_path: Path) -> None:
    """Tests _load_pickle with a pickle file containing objects of incorrect types."""
    wrong_types_file = tmp_path / "wrong_types.pkl"
    with wrong_types_file.open("wb") as f:
        pickle.dump({"vectoriser": "not a vectorizer", "estimator": 123}, f)

    with pytest.raises(RuntimeError, match="objects of unexpected types"):
        _load_pickle(wrong_types_file)


def test_get_model_loads_and_caches(temp_model_path: Path) -> None:
    """Tests that _get_model loads the model and caches the result."""
    with patch(
        "src.classification.model._load_pickle", wraps=_load_pickle
    ) as mock_loader:
        # First call - should load
        container1 = _get_model(temp_model_path)
        mock_loader.assert_called_once_with(temp_model_path)

        # Second call - should use cache
        mock_loader.reset_mock()
        container2 = _get_model(temp_model_path)
        mock_loader.assert_not_called()

        assert container1 is container2  # Ensure same instance is returned


def test_get_model_file_not_found_propagates(tmp_path: Path) -> None:
    """Tests that FileNotFoundError from _load_pickle propagates through _get_model."""
    non_existent_path = tmp_path / "not_here.pkl"
    with pytest.raises(FileNotFoundError):
        _get_model(non_existent_path)


@pytest.mark.asyncio  # predict itself isn't async, but tests might run in async context
async def test_predict_success(temp_model_path: Path) -> None:
    """Tests the main predict function with a valid model."""
    # Use patch to control the model loaded by predict's internal _get_model
    with patch("src.classification.model._get_model") as mock_getter:
        # Configure the mock _get_model to return a model container
        mock_container = MagicMock(spec=_ModelContainer)
        mock_container.predict.return_value = ("predicted_label", 0.95)
        mock_getter.return_value = mock_container

        label, probability = predict("some text")

        assert label == "predicted_label"
        assert probability == pytest.approx(0.95)
        mock_getter.assert_called_once()  # Check _get_model was called
        mock_container.predict.assert_called_once_with("some text")


@pytest.mark.asyncio
async def test_predict_empty_text() -> None:
    """Tests predict function with empty or whitespace-only input text."""
    label, probability = predict("")
    assert label is None
    assert probability is None

    label, probability = predict("   \n\t ")
    assert label is None
    assert probability is None


@pytest.mark.asyncio
async def test_predict_model_not_available(tmp_path: Path) -> None:
    """Tests predict function when the model cannot be loaded."""
    non_existent_path = tmp_path / "no_model_here.pkl"
    # Patch _get_model to simulate failure
    with patch(
        "src.classification.model._get_model",
        side_effect=FileNotFoundError("Model missing"),
    ):
        with pytest.raises(ModelNotAvailableError, match="Model missing"):
            predict("some text")
