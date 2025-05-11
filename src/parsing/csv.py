from __future__ import annotations

import asyncio
from io import BytesIO

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


async def extract_text_from_csv(file: UploadFile) -> str:
    """
    Extract text content from a CSV file, converting it to a readable format.

    Args:
        file: The uploaded CSV file

    Returns:
        Extracted text content as a string
    """
    await file.seek(0)
    content = await file.read()

    def _worker(csv_content: bytes) -> str:
        try:
            csv_buffer = BytesIO(csv_content)
            df = pd.read_csv(csv_buffer)

            if df.empty:
                return ""

            header_row = " ".join(df.columns.to_list())

            data_rows = []
            for _, row in df.iterrows():
                data_rows.append(" ".join(str(v) for v in row.to_list()))

            return header_row + "\n" + "\n".join(data_rows)

        except (EmptyDataError, ParserError):
            # Fallback to raw text for malformed CSV
            try:
                return csv_content.decode("utf-8", errors="replace")
            except Exception:
                return ""
        except Exception:
            return ""

    return await asyncio.to_thread(_worker, content)
