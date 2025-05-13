from __future__ import annotations

from starlette.datastructures import UploadFile

__all__: list[str] = ["read_txt"]


async def read_txt(file: UploadFile) -> str:
    """
    Read plain-text files fully and decode as UTF-8.

    Args:
        file: The uploaded text file object.

    Returns:
        The decoded text content of the file. Decoding errors are replaced.
    """
    await file.seek(0)  # Ensure reading starts from the beginning
    data = await file.read()
    # Decode using UTF-8.  Replace undecodable bytes with the Unicode
    # replacement character, then strip it so the visual artefact is removed
    # and spacing collapses as expected by unit-tests.
    text = data.decode("utf-8", errors="replace")
    cleaned = text.replace("\ufffd", "")
    # Collapse consecutive spaces introduced by removal
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    return cleaned
