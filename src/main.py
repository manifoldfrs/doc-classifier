"""
Main FastAPI application module.

This module initializes the FastAPI application, registers middlewares,
routers, and event handlers.
"""

from fastapi import FastAPI, APIRouter

# Create FastAPI app instance
app = FastAPI(
    title="Heron File Classifier",
    description="API for classifying files based on content and metadata",
    version="0.1.0",
)

# Create API router for versioned endpoints
api_router = APIRouter()

# Include routers from endpoints modules
# This will be populated when we implement the endpoints in future steps
# from src.api.v1.endpoints.classification import router as classification_router
# api_router.include_router(classification_router, prefix="/api/v1")

# Include the API router in the main app
app.include_router(api_router)


@app.get("/")
def root():
    """
    Root endpoint to check if the API is running.

    Returns:
        dict: A simple message indicating the API is running.
    """
    return {"message": "Heron File Classifier API is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
