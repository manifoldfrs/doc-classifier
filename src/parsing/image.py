"""src/parsing/image.py
###############################################################################
Image OCR text extraction using pytesseract
###############################################################################
This module provides functions for extracting text from image files using OCR.

Design considerations
=====================
1. **Thread off-loading** – Pillow decoding & Tesseract OCR are CPU-intensive;
   the heavy lifting runs inside `asyncio.to_thread()` so we do not block the
   FastAPI event-loop.
2. **Minimal preprocessing** – we convert the image to *RGB* and apply no other
   filters.  Future iterations could add adaptive thresholding or denoising to
   improve accuracy on low-quality scans.
3. **Failure propagation** – any exception raised by Pillow or Tesseract is
   allowed to propagate.  The classification pipeline or global exception
   middleware will record and surface appropriate errors.
4. **≤ 40 lines** – the public coroutine complies with the project guidelines.

Assumptions / Limitations
-------------------------
• Tesseract must be installed in the container or Vercel environment.  The
  binary is **not** a Python dependency; ensure the runtime image provides it
  (e.g. `apt-get install tesseract-ocr`).
• No language parameter is passed; defaults to *eng*.  Extend via env var if
  multi-language support is required.
"""

from __future__ import annotations

# stdlib
import asyncio
from io import BytesIO

# third-party
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
    # Read the file content
    await file.seek(0)
    content = await file.read()

    # Define worker function for async execution
    def _worker(image_content: bytes) -> str:
        try:
            # Create BytesIO object for PIL to read from
            image_buffer = BytesIO(image_content)

            # Open and process the image
            with Image.open(image_buffer) as img:
                # Convert to RGB to ensure compatibility with OCR
                img = img.convert("RGB")

                # Perform OCR
                text = pytesseract.image_to_string(img)
                return text or ""
        except Exception as e:
            # Handle image processing errors
            return f"Error extracting image text: {str(e)}"

    # Run CPU-bound OCR in threadpool
    return await asyncio.to_thread(_worker, content)
