from __future__ import annotations

from werkzeug.datastructures import FileStorage


def classify_file(file: FileStorage) -> str:
    # FileStorage.filename is Optional[str]; guard against *None* for strict typing.
    filename = (file.filename or "").lower()
    # file_bytes = file.read()

    if "drivers_license" in filename:
        return "drivers_licence"

    if "bank_statement" in filename:
        return "bank_statement"

    if "invoice" in filename:
        return "invoice"

    return "unknown file"
