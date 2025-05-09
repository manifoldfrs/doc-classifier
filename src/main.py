"""
Main FastAPI application module.

This module initializes the FastAPI application, registers middlewares,
routers, and event handlers.
"""

import logging
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.core.logging_config import setup_logging, LoggingMiddleware
from src.api.v1.endpoints.classification import (
    router as classification_router_v1,
)

# Setup logging
# Configure logging based on settings (DEBUG or INFO)
setup_logging(log_level="DEBUG" if settings.DEBUG else "INFO")
logger = logging.getLogger(__name__)


# Create FastAPI app instance
app = FastAPI(
    title=settings.APP_NAME,
    description="API for classifying files based on content and metadata.",
    version="0.1.0",  # Consider making this dynamic or part of settings
    debug=settings.DEBUG,
    # openapi_url="/api/v1/openapi.json" # Example if you want to customize openapi path
    # docs_url="/api/v1/docs" # Example for customizing docs
)

# Set up CORS middleware
# WARNING: Allowing all origins ("*") is insecure for production.
# Restrict this to specific domains in a production environment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else [],  # More restrictive in prod
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
        "PUT",
        "DELETE",
        "OPTIONS",
    ],  # Specify allowed methods
    allow_headers=["*"],  # Specify allowed headers or be more restrictive
)

# Add logging middleware
# This custom middleware logs request and response details.
app.middleware("http")(LoggingMiddleware())


# Create API router for versioned endpoints
# Using APIRouter helps organize endpoints, especially with versioning.
api_v1_router = APIRouter(prefix="/api/v1")

# Include routers from specific endpoint modules
api_v1_router.include_router(
    classification_router_v1,
    prefix="/classification",  # Prefix for all routes in classification_router_v1
    tags=["Classification"],  # Tag for OpenAPI documentation grouping
)

# Include the versioned API router in the main app
app.include_router(api_v1_router)


@app.on_event("startup")
async def startup_event() -> None:
    """
    Execute startup tasks for the application.
    This can include initializing database connections, loading ML models, etc.
    """
    logger.info(f"Starting {settings.APP_NAME} API...")
    # Example: Initialize database connections
    # await db_manager.connect()
    # Example: Load ML models
    # ml_model_loader.load_models()
    logger.info(f"{settings.APP_NAME} API started successfully.")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """
    Execute shutdown tasks for the application.
    This is important for gracefully releasing resources like database connections.
    """
    logger.info(f"Shutting down {settings.APP_NAME} API...")
    # Example: Close database connections
    # await db_manager.disconnect()
    logger.info(f"{settings.APP_NAME} API shut down gracefully.")


@app.get(
    "/", include_in_schema=False
)  # Exclude from OpenAPI docs if it's just a health check
async def root() -> dict[str, str]:
    """
    Root endpoint to check if the API is running.

    Returns:
        A simple dictionary message indicating the API status.
    """
    logger.debug("Root endpoint '/' accessed.")
    return {"message": f"{settings.APP_NAME} API is running."}


# This block allows running the app directly with Uvicorn for development.
# In production, a process manager like Gunicorn or Uvicorn with multiple workers
# would typically be used.
if __name__ == "__main__":
    import uvicorn

    # Note: Reload should be False in production
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",  # Listen on all available network interfaces
        port=8000,  # Standard port for web services
        reload=settings.DEBUG,  # Enable auto-reload in debug mode
        log_level=logging.getLevelName(logger.getEffectiveLevel()).lower(),
    )
