from __future__ import annotations

import os

from dotenv import load_dotenv  # type: ignore[import-not-found]
from flask import Flask, Response, jsonify, request  # type: ignore[import-not-found]

from src.classifier import classify_file

load_dotenv()

app = Flask(__name__)


ALLOWED_EXTENSIONS = {
    ext.strip().lower()
    for ext in (
        os.getenv("ALLOWED_EXTENSIONS")
        or "pdf,docx,doc,xlsx,xlsb,xls,csv,jpg,jpeg,png,txt,md,xml,json,html,eml"
    ).split(",")
    if ext.strip()
}


def allowed_file(filename: str) -> bool:
    """Return ``True`` if *filename* has an allowed extension.

    The check is case-insensitive and expects a period separating the base name
    and extension (e.g., ``"document.pdf"``).
    """

    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/classify_file", methods=["POST"])
def classify_file_route() -> Response:  # noqa: D401

    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed: {file.filename}"}), 400

    file_class = classify_file(file)
    return jsonify({"file_class": file_class}), 200


if __name__ == "__main__":
    app.run(debug=True)
