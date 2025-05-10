from __future__ import annotations

# third-party
from werkzeug.datastructures import FileStorage  # type: ignore[import-not-found]


def classify_file(file: FileStorage) -> str:
    filename = file.filename.lower()
    # file_bytes = file.read()

    if "drivers_license" in filename:
        return "drivers_licence"

    if "bank_statement" in filename:
        return "bank_statement"

    if "invoice" in filename:
        return "invoice"

    return "unknown file"
