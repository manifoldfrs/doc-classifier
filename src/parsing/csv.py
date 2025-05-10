"""src/parsing/csv.py
###############################################################################
CSV text-extraction adapter
###############################################################################
This helper converts a **comma-separated values** document (or other delimiter
handled by *pandas*) supplied via a FastAPI `UploadFile` into a raw UTF-8
string.  The returned text is *not* cleaned or normalised – that responsibility
belongs to the classification pipeline.

Design considerations
=====================
1. **Async-friendly** – `pandas.read_csv` is CPU/IO-bound and synchronous.  We
   off-load the entire parsing step to a thread via `asyncio.to_thread()` so the
   event-loop remains responsive.
2. **Robust to encoding issues** – if pandas fails due to an unknown
   encoding/parsing error we gracefully fall back to a *naïve* UTF-8 decode of
   the raw bytes so that higher stages still receive *some* text.
3. **≤ 40 lines per public function** – adheres to project coding rules.
4. **No catch-all in the hot-path** – we explicitly handle known pandas
   exceptions; unexpected errors bubble up to FastAPI's global handler.

Limitations / Future work
-------------------------
• The fallback decoder currently assumes UTF-8 input.  In production we could
  employ `chardet`/`cchardet` or `charset-normalizer` to auto-detect encodings.
• Delimiter detection is delegated to pandas' Python engine which is slower but
  more flexible.  Performance tuning may be required for huge (>10 MB) files in
  future iterations when the service adds streaming CSV support.
"""

from __future__ import annotations

# stdlib
import asyncio
from io import BytesIO

# third-party
import pandas as pd
import structlog
from pandas.errors import EmptyDataError, ParserError
from starlette.datastructures import UploadFile

__all__: list[str] = [
    "extract_text_from_csv",
]

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers – kept private to avoid export noise
# ---------------------------------------------------------------------------


def _dataframe_to_text(df: pd.DataFrame) -> str:  # noqa: D401
    """Flatten a **pandas** DataFrame into a whitespace-separated string.

    Each row becomes one line; columns are joined by a single space.  `NaN`
    values are replaced with empty strings for stability.
    """

    header: str = " ".join(map(str, df.columns.to_list()))
    lines: list[str] = [header]
    for _, row in df.iterrows():
        # Convert all cells to str while replacing NaNs with empty strings
        cells = ["" if pd.isna(c) else str(c) for c in row.to_list()]
        lines.append(" ".join(cells))
    return "\n".join(lines)


async def extract_text_from_csv(file: UploadFile) -> str:  # noqa: D401
    """Return **plain-text** extracted from a CSV *file*.

    The coroutine prioritises `pandas.read_csv` due to its flexible parsing.
    When pandas raises `EmptyDataError` or `ParserError` we fall back to a raw
    decode of the bytes so that downstream stages are not starved of input.

    Parameters
    ----------
    file:
        The upload object supplied by FastAPI.

    Returns
    -------
    str
        Textual representation of the CSV suitable for TF-IDF vectorisation.
    """

    # Ensure file pointer at start
    await file.seek(0)
    csv_bytes: bytes = await file.read()

    def _parse_with_pandas() -> str:
        # The *python* engine handles autodetection of delimiters better at the
        # expense of speed – acceptable given ≤ 10 MB file size.
        df: pd.DataFrame = pd.read_csv(
            BytesIO(csv_bytes), engine="python", dtype=str, na_filter=False
        )
        return _dataframe_to_text(df)

    text: str | None = None
    try:
        text = await asyncio.to_thread(_parse_with_pandas)
    except (EmptyDataError, ParserError) as exc:
        logger.warning("csv_pandas_failed", filename=file.filename, error=str(exc))

    # ------------------------------------------------------------------
    # Fallback – naïve UTF-8 decode with error replacement
    # ------------------------------------------------------------------
    if text is None or not text.strip():
        text = csv_bytes.decode("utf-8", errors="replace")

    logger.debug("csv_text_extracted", filename=file.filename, characters=len(text))
    return text
