from __future__ import annotations

import asyncio
from io import BytesIO

import pytesseract
import structlog
from PIL import Image  # Pillow must be present in runtime
from starlette.datastructures import UploadFile

__all__: list[str] = [
    "extract_text_from_image",
]

logger = structlog.get_logger(__name__)


async def extract_text_from_image(file: UploadFile) -> str:
    """
    Extract text from an image file using OCR with Tesseract.

    Args:
        file: The uploaded image file

    Returns:
        Extracted text content as a string
    """
    await file.seek(0)
    content = await file.read()

    def _worker(image_content: bytes) -> str:
        try:
            image_buffer = BytesIO(image_content)

            with Image.open(image_buffer) as img:
                img = img.convert("RGB")

                text = pytesseract.image_to_string(img)
                return text or ""
        except Exception as e:
            return f"Error extracting image text: {str(e)}"

    return await asyncio.to_thread(_worker, content)
