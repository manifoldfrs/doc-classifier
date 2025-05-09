"""
Classification schemas for the file classifier API.

This module defines Pydantic models for classification request and response validation.
These models are used by the API endpoints to ensure data consistency and provide
clear contracts for API consumers.
"""

from pydantic import BaseModel, Field


class ClassificationResponse(BaseModel):
    """
    Pydantic model for the classification response.

    Attributes:
        file_class: The predicted class of the file.
        confidence: The confidence score of the classification (0.0 to 1.0).
    """

    file_class: str = Field(..., description="The predicted class of the file.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="The confidence score of the classification (0.0 to 1.0).",
    )


# Add other relevant schemas here as the project evolves, for example:
# class FileUploadRequest(BaseModel):
#     """
#     Pydantic model for file upload metadata if needed separately.
#     Currently, FastAPI handles File Uploads directly.
#     """
#     filename: str
#     content_type: str

# class ClassificationErrorResponse(BaseModel):
#     """
#     Pydantic model for error responses during classification.
#     """
#     detail: str
#     error_code: Optional[str] = None
