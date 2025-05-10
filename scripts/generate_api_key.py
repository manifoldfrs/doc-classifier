#!/usr/bin/env python
###############################################################################
# scripts/generate_api_key.py
# -----------------------------------------------------------------------------
# Utility script that generates cryptographically-random API keys compatible
# with the HeronAI demo service.  Keys are **UUIDv4** values rendered in their
# canonical 32-char *hex* representation (no dashes) so they fit comfortably
# in HTTP headers and environment variables.
#
# Usage
# -----
#   python scripts/generate_api_key.py            # prints one key
#   python scripts/generate_api_key.py --count 5  # prints five keys
#
# The script does *not* mutate any files – it only prints to stdout so callers
# can redirect output or copy-paste the keys into their `.env` under the
# `ALLOWED_API_KEYS` variable.
#
# Rationale
# =========
# • Keeping the helper in *scripts/* avoids adding runtime dependencies.
# • UUIDv4 provides 122 bits of entropy – sufficient for demo-grade secrets.
###############################################################################

from __future__ import annotations

import argparse
import sys
import uuid
from typing import List

__all__: list[str] = []  # script – no public API


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
