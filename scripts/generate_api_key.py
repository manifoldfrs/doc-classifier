#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import uuid
from typing import List

__all__: list[str] = []


def _parse_args() -> argparse.Namespace:  # noqa: D401 – CLI helper
    parser = argparse.ArgumentParser(
        description="Generate random API keys for the HeronAI demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of keys to generate.",
    )
    return parser.parse_args()


def _generate_keys(n: int) -> List[str]:  # noqa: D401 – internal helper
    if n <= 0:
        raise ValueError("count must be a positive integer")
    return [uuid.uuid4().hex for _ in range(n)]


def main() -> None:  # noqa: D401 – entry-point
    args = _parse_args()
    try:
        keys = _generate_keys(args.count)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    for key in keys:
        print(key)


if __name__ == "__main__":  # pragma: no cover – CLI only
    main()
