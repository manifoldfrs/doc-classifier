import importlib
import os
import sys

import pytest


@pytest.mark.usefixtures("monkeypatch")
class TestConfigEnvPreprocessing:
    """Cover *src.core.config* environment preprocessing logic executed at import-time."""

    def test_env_vars_are_normalised_to_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Comma-separated ALLOWED_* values should be converted to JSON on module import."""

        # Arrange – set comma-separated env vars
        monkeypatch.setenv("ALLOWED_API_KEYS", "k1, k2")
        monkeypatch.setenv("ALLOWED_EXTENSIONS", "pdf, txt")

        # Remove cached module (if any) so import executes top-level preprocessing again
        sys.modules.pop("src.core.config", None)

        # Act – re-import the module
        cfg = importlib.import_module("src.core.config")

        # Assert – env vars were rewritten as JSON arrays by the import-time hook
        assert os.environ["ALLOWED_API_KEYS"].startswith("[")
        assert os.environ["ALLOWED_EXTENSIONS"].startswith("[")

        settings = cfg.get_settings()  # Fresh settings instance
        assert settings.allowed_api_keys == ["k1", "k2"]
        assert settings.allowed_extensions == {"pdf", "txt"}

    def test_invalid_threshold_validation(self) -> None:
        """early_exit_confidence lower than confidence_threshold must raise."""
        cfg = importlib.import_module("src.core.config")
        Settings = cfg.Settings  # type: ignore[attr-defined]

        with pytest.raises(ValueError):
            Settings(confidence_threshold=0.8, early_exit_confidence=0.5)

    def test_coerce_allowed_extensions_json(self) -> None:
        """JSON string should be coerced into a normalised set."""
        cfg = importlib.import_module("src.core.config")
        Settings = cfg.Settings  # type: ignore[attr-defined]

        s = Settings(allowed_extensions='["PDF", "JPG"]')
        assert s.allowed_extensions == {"pdf", "jpg"}
        # is_extension_allowed should respect leading dots & case
        assert s.is_extension_allowed(".PDF") is True
        assert s.is_extension_allowed("png") is False
