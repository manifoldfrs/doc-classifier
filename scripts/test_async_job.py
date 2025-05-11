from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import requests

DEFAULT_DIR = Path("files")
ASYNC_THRESHOLD = 11  # backend enqueues when >10


def _parse_args() -> argparse.Namespace:  # noqa: D401
    parser = argparse.ArgumentParser(
        description="Upload >10 files and poll async /v1/jobs endpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL where FastAPI is running (no trailing slash).",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_DIR,
        help="Directory holding sample docs (ignored when --files is used).",
    )
    parser.add_argument(
        "--glob",
        default="*",
        help="Glob pattern to filter files inside --dir.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=ASYNC_THRESHOLD + 1,
        help="Number of files to upload (must exceed backend threshold).",
    )
    parser.add_argument(
        "--poll",
        dest="poll_interval",
        type=float,
        default=1.0,
        help="Initial polling interval in seconds (exponential back-off).",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default="",
        help="Optional value for x-api-key header.",
    )
    return parser.parse_args()


def _collect_files(directory: Path, pattern: str, count: int) -> List[Path]:
    if not directory.is_dir():
        print(f"❌ {directory} is not a directory", file=sys.stderr)
        sys.exit(1)
    files = sorted(directory.glob(pattern))[:count]
    if len(files) < count:
        print(
            f"❌ Only {len(files)} files matched pattern – need {count}+.",
            file=sys.stderr,
        )
        sys.exit(1)
    return files


def _build_multipart(paths: List[Path]):
    return [("files", (p.name, p.read_bytes())) for p in paths]


def main() -> None:  # noqa: D401
    args = _parse_args()

    endpoint_files = f"{args.url.rstrip('/')}/v1/files"
    endpoint_jobs = f"{args.url.rstrip('/')}/v1/jobs"  # /{job_id} later

    paths = _collect_files(args.dir, args.glob, args.count)
    print(f"▶ Uploading {len(paths)} files → {endpoint_files}")

    headers = {"x-api-key": args.api_key} if args.api_key else {}
    resp = requests.post(endpoint_files, files=_build_multipart(paths), headers=headers)

    if resp.status_code not in {202}:
        print(
            f"❌ Expected 202 but got {resp.status_code}\n{resp.text}", file=sys.stderr
        )
        sys.exit(1)

    job_info = resp.json()
    job_id = job_info["job_id"]
    print(f"✔ Job accepted – id={job_id}\n")

    # ----------------------   polling   ----------------------------------
    delay = args.poll_interval
    while True:
        time.sleep(delay)
        r = requests.get(f"{endpoint_jobs}/{job_id}", headers=headers)
        if r.status_code != 200:
            print(f"❌ Polling error HTTP {r.status_code}: {r.text}", file=sys.stderr)
            sys.exit(1)
        payload = r.json()
        status = payload.get("status")
        print(f"⏳ status={status}  (next check {delay:.1f}s)")
        if status == "done":
            print("\n=== FINAL RESULT ===")
            print(json.dumps(payload, indent=2))
            break
        delay = min(delay * 1.5, 5.0)  # cap at 5s


if __name__ == "__main__":  # pragma: no cover
    main()
