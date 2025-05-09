"""
Configuration module for the application.

This module defines the application settings using Pydantic for validation
and loading from environment variables.
"""

from typing import Set
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings.

    Attributes:
        APP_NAME: Name of the application.
        DEBUG: Whether to run in debug mode.
        ALLOWED_API_KEYS: Set of valid API keys for authentication.
        ALLOWED_EXTENSIONS: Set of allowed file extensions.
    """

    APP_NAME: str = "Heron File Classifier"
    DEBUG: bool = False
    ALLOWED_API_KEYS: Set[str] = {"your_default_secret_key"}
    ALLOWED_EXTENSIONS: Set[str] = {"pdf", "png", "jpg"}

    class Config:
        """Pydantic config for Settings."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create a settings instance
settings = Settings()
