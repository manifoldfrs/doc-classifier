"""
Main FastAPI application module.

This module initializes the FastAPI application, registers middlewares,
routers, and event handlers.
"""

import logging
from fastapi import FastAPI, APIRouter, Depends
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.core.logging_config import setup_logging, LoggingMiddleware

# Setup logging
setup_logging(log_level="DEBUG" if settings.DEBUG else "INFO")
logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI(
    title="Heron File Classifier",
    description="API for classifying files based on content and metadata",
    version="0.1.0",
    debug=settings.DEBUG,
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware
app.middleware("http")(LoggingMiddleware())

# Create API router for versioned endpoints
api_router = APIRouter()

# Include routers from endpoints modules
# This will be populated when we implement the endpoints in future steps
# from src.api.v1.endpoints.classification import router as classification_router
# api_router.include_router(classification_router, prefix="/api/v1")

# Include the API router in the main app
app.include_router(api_router)


@app.on_event("startup")
async def startup_event():
    """Execute startup tasks."""
    logger.info("Starting Heron File Classifier API")


@app.on_event("shutdown")
async def shutdown_event():
    """Execute shutdown tasks."""
    logger.info("Shutting down Heron File Classifier API")


@app.get("/")
def root():
    """
    Root endpoint to check if the API is running.

    Returns:
        dict: A simple message indicating the API is running.
    """
    logger.debug("Root endpoint accessed")
    return {"message": "Heron File Classifier API is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
