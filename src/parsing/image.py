"""src/parsing/image.py
###############################################################################
Image OCR text-extraction adapter
###############################################################################
This adapter extracts textual content from **raster images** (JPEG/PNG) using
*Tesseract OCR* via the `pytesseract` binding.  It converts the user-supplied
`UploadFile` into a Pillow `Image` instance and delegates recognition to
`pytesseract.image_to_string`.

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


async def extract_text_from_image(file: UploadFile) -> str:  # noqa: D401
    """Return textual content obtained via **OCR** on an image *file*.

    Parameters
    ----------
    file:
        The uploaded raster image wrapped by FastAPI.

    Returns
    -------
    str
        OCR-extracted string (may be empty if no text detected).
    """

    # Reset pointer in case validators have already read from the stream
    await file.seek(0)
    image_bytes: bytes = await file.read()

    def _ocr_worker() -> str:
        # Pillow handles format detection internally
        with Image.open(BytesIO(image_bytes)) as img:
            # Ensure consistent colour mode for tesseract
            img_rgb = img.convert("RGB")
            return pytesseract.image_to_string(img_rgb) or ""

    text: str = await asyncio.to_thread(_ocr_worker)

    logger.debug("image_text_extracted", filename=file.filename, characters=len(text))
    return text
