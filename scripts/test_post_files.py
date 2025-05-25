from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import requests

DEFAULT_FILES_DIR = Path("files")  # repository sample docs


def _parse_args() -> argparse.Namespace:  # noqa: D401
    parser = argparse.ArgumentParser(
        description="Upload sample files to the /v1/files endpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000/v1/files",
        help="Full URL to the /v1/files endpoint.",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default="",
        help="Value for the x-api-key header (leave blank if auth disabled).",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_FILES_DIR,
        help=(
            "Directory *or single file* to upload.  When a file path is "
            "provided the --glob and --max arguments are ignored."
        ),
    )
    parser.add_argument(
        "--glob",
        default="*",
        help="Glob pattern to filter files in the directory (e.g. '*.pdf').",
    )
    parser.add_argument(
        "--max",
        dest="max_files",
        type=int,
        default=10,
        help="Maximum number of files to send in one request.",
    )
    return parser.parse_args()


def _collect_files(path: Path, pattern: str, limit: int) -> List[Path]:  # noqa: D401
    """Return a list of file paths to upload.

    The helper supports two modes:
    1. **Directory mode** – *path* is a directory; return up to *limit* files
       matching *pattern*.
    2. **File mode** – *path* is a file; return a single-element list.
    """

    if path.is_file():
        return [path]

    if not path.is_dir():
        print(f"Error: {path} is neither a file nor a directory", file=sys.stderr)
        sys.exit(1)

    files = sorted(path.glob(pattern))[:limit]
    if not files:
        print(f"No files matched '{pattern}' in {path}", file=sys.stderr)
        sys.exit(1)
    return files


def _build_multipart(files: List[Path]):  # noqa: D401
    """Build a ``files`` dict suitable for requests.post(..., files=...)."""

    multipart = []
    for path in files:
        multipart.append(("files", (path.name, path.read_bytes())))
    return multipart


def main() -> None:  # noqa: D401
    args = _parse_args()

    file_paths: List[Path] = _collect_files(args.dir, args.glob, args.max_files)

    print(f"Uploading {len(file_paths)} file(s) → {args.url}\n")

    headers = {"x-api-key": args.api_key} if args.api_key else {}

    response = requests.post(
        args.url, files=_build_multipart(file_paths), headers=headers
    )

    if response.status_code != 200:
        print(f"❌ HTTP {response.status_code}\n{response.text}", file=sys.stderr)
        sys.exit(1)

    try:
        payload = response.json()
        print(json.dumps(payload, indent=2))
    except ValueError:
        print("❌ Response is not valid JSON:")
        print(response.text)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
