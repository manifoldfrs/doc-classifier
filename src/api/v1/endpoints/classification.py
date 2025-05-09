"""
Classification endpoints for the file classifier API.

This module defines the API endpoints for file classification operations,
including synchronous and asynchronous file classification.
"""

import logging
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, status

from src.api.v1.schemas.classification import ClassificationResponse
from src.services.classification_service import ClassificationService
from src.api.security import get_api_key
from src.core.config import settings

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Create an API router
router = APIRouter()

# Instantiate services that the endpoints will use.
# In a more complex application with dependency injection frameworks,
# this might be handled differently (e.g., using FastAPI's Depends for services).
classification_service = ClassificationService()


def allowed_file(filename: str | None) -> bool:
    """
    Checks if the uploaded file has an allowed extension.

    Args:
        filename: The name of the file.

    Returns:
        True if the file extension is in the allowed set, False otherwise.
    """
    if not filename:
        return False
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in settings.ALLOWED_EXTENSIONS
    )


@router.post(
    "/classify_file",
    response_model=ClassificationResponse,
    summary="Synchronously classify a single file",
    description="Upload a file (PDF, PNG, JPG) to classify it based on its filename. "
    "This is a basic synchronous endpoint. For larger files or "
    "more complex processing, asynchronous endpoints are recommended.",
    dependencies=[Depends(get_api_key)],
)
async def classify_file_sync(
    file: UploadFile = File(
        ..., description="The file to be classified (PDF, PNG, JPG)."
    )
) -> ClassificationResponse:
    """
    Synchronous endpoint to classify an uploaded file.

    It performs basic validation on the file (presence, name, allowed type)
    and then uses the `ClassificationService` to determine the file class
    based on filename heuristics.

    Args:
        file: The uploaded file, injected by FastAPI.
        api_key: The API key for authentication, injected by `get_api_key` dependency.

    Returns:
        A `ClassificationResponse` Pydantic model containing the `file_class`
        and `confidence` score.

    Raises:
        HTTPException:
            - 400 (Bad Request): If no file is provided, filename is empty, or
                                 file type is not allowed.
            - 500 (Internal Server Error): If an unexpected error occurs during
                                         classification.
    """
    logger.info(
        f"Received synchronous classification request for file: {file.filename}"
    )

    if not file:
        logger.warning("No file provided in the request.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file part in the request.",
        )
    if not file.filename:
        logger.warning("Empty filename provided.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No selected file."
        )

    if not allowed_file(file.filename):
        logger.warning(
            f"File type not allowed for file: {file.filename}. "
            f"Allowed types: {settings.ALLOWED_EXTENSIONS}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File type not allowed. Allowed types are: "
                f"{', '.join(settings.ALLOWED_EXTENSIONS)}"
            ),
        )

    try:
        # Use the classification service to classify the file by filename
        # This is a placeholder for more sophisticated classification later
        file_class, confidence = classification_service.classify_file_by_filename(file)
        logger.info(
            f"File '{file.filename}' classified as '{file_class}' with confidence {confidence}."
        )
        return ClassificationResponse(file_class=file_class, confidence=confidence)
    except Exception as e:
        logger.error(
            f"Error during synchronous classification of file '{file.filename}': {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during file classification.",
        )


# Future asynchronous endpoints will be added here:
# @router.post("/classify_file_async", ...)
# async def classify_file_async(...): ...

# @router.get("/classification_status/{task_id}", ...)
# async def get_classification_status(task_id: str, ...): ...
