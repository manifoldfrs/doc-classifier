#!/usr/bin/env python
###############################################################################
# scripts/generate_synthetic.py
# -----------------------------------------------------------------------------
# Synthetic document generator (Implementation Plan – Step 8.1)
#
# This CLI utility produces a *corpus* of **synthetic documents** across the
# primary labels required by the HeronAI demo service.  The generated artefacts
# are intended for **manual experimentation, model prototyping**, and pipeline
# demonstrations.  They are *not* used directly by the runtime service which
# operates exclusively on user-supplied uploads.
#
# Key characteristics
# ===================
# 1. **Multi-format output** – depending on the *label* it emits `.txt`, `.csv`,
#    or `.png` files so the parsing layer is covered.
# 2. **Lightweight dependencies** – utilises only *Faker*, *pandas*, and
#    *Pillow* – all of which are already part of the project requirements.
# 3. **Configurable via CLI** – callers can tweak the number of samples, random
#    seed, and output directory.
# 4. **≤ 40 lines per function** – adheres to repo engineering rules.
# 5. **Explicit typing & no broad excepts** – `mypy --strict` compliant.
#
# Example
# -------
#     python scripts/generate_synthetic.py --count 500 --out datasets/synthetic
#
# The command prints a summary table and writes ~500 files spread across
#    datasets/synthetic/<label>/.
###############################################################################

from __future__ import annotations

# stdlib
import argparse
import random
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# third-party
from faker import Faker  # type: ignore
from PIL import Image, ImageDraw, ImageFont  # type: ignore

__all__: List[str] = []  # script – no public API

###############################################################################
# Constants & templates
###############################################################################

faker: Faker = Faker()

_LABELS: Tuple[str, ...] = (
    "invoice",
    "bank_statement",
    "financial_report",
    "drivers_licence",
    "contract",
    "email",
    "form",
)

# Simple sentence templates reused from *train_model.py* so the statistical
# model and file generator share a coherent vocabulary.
_TEMPLATES: Dict[str, List[str]] = {
    "invoice": [
        "Invoice #{num} – amount due {amount} {currency}",
        "Tax invoice for order {num}",
        "Payment reminder: Invoice {num}",
    ],
    "bank_statement": [
        "Bank statement for account {num}",
        "Statement period {month} – ending balance {amount} {currency}",
        "Transaction list – acct {num}",
    ],
    "financial_report": [
        "Annual financial report {year}",
        "Q{q} performance report",
        "Balance sheet summary",
    ],
    "drivers_licence": [
        "Driver licence – {state} – ID {num}",
        "DL #{num} issued to {name}",
    ],
    "contract": [
        "Service contract between {company1} and {company2}",
        "Employment agreement – {company1}",
        "Contract addendum #{num}",
    ],
    "email": [
        "From: {name}@example.com\nTo: support@example.com\nSubject: Order {num}",
        "Subject: Inquiry about invoice {num}",
        "From: hr@{company1}.com\nSubject: Offer letter",
    ],
    "form": [
        "Application form – {company1}",
        "Registration form for event {name}",
        "Order form #{num}",
    ],
}

###############################################################################
# Template renderer & content builders – kept small to satisfy line limits
###############################################################################


def _render(template: str) -> str:
    """Return *template* with faker placeholders substituted."""

    return (
        template.replace("{num}", str(faker.random_number(digits=6)))
        .replace("{amount}", f"{faker.pydecimal(left_digits=4, right_digits=2)}")
        .replace("{currency}", faker.currency_code())
        .replace("{month}", faker.month_name())
        .replace("{year}", str(faker.year()))
        .replace("{q}", str(faker.random_int(min=1, max=4)))
        .replace("{state}", faker.state())
        .replace("{name}", faker.first_name())
        .replace("{company1}", faker.company().replace(" ", ""))
        .replace("{company2}", faker.company().replace(" ", ""))
    )


def _build_text(label: str) -> str:
    """Return a multi-line **text blob** representative of *label*."""

    lines: List[str] = []
    for _ in range(random.randint(3, 8)):
        tmpl: str = random.choice(_TEMPLATES[label])
        lines.append(_render(tmpl))
    # Wrap to 80 chars for nicer plain-text docs
    return "\n".join(textwrap.fill(line, 80) for line in lines)


def _write_txt(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, text: str) -> None:
    # Pretend bank statements have date/desc/amount columns
    rows: List[Tuple[str, str, str]] = []
    for _ in range(random.randint(5, 15)):
        rows.append(
            (
                faker.date_this_year().isoformat(),
                faker.sentence(nb_words=5),
                f"{faker.pydecimal(left_digits=3, right_digits=2)}",
            )
        )
    df = pd.DataFrame(rows, columns=["date", "description", "amount"])
    df.to_csv(path, index=False)


def _write_png(path: Path, text: str) -> None:
    # Create a simple white canvas and draw text using PIL.  No external font
    # dependencies – default bitmap font is sufficient for OCR.
    width, height = 800, 600
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    margin, offset = 10, 10
    for line in text.split("\n"):
        draw.text((margin, offset), line, fill="black", font=font)
        try:
            # Pillow >= 8.0 preferred API
            bbox = draw.textbbox((margin, offset), line, font=font)
            line_height: int = bbox[3] - bbox[1]
        except AttributeError:  # Pillow < 8.0 fallback
            line_height = draw.textsize(line, font=font)[1]  # type: ignore[attr-defined]
        offset += line_height + 4  # small vertical spacing
    img.save(path, format="PNG")


###############################################################################
# Dispatch table – label → (extension, writer-func)
###############################################################################

_WRITERS: Dict[str, Tuple[str, callable[[Path, str], None]]] = {
    "invoice": ("txt", _write_txt),
    "bank_statement": ("csv", _write_csv),
    "financial_report": ("txt", _write_txt),
    "drivers_licence": ("png", _write_png),
    "contract": ("txt", _write_txt),
    "email": ("txt", _write_txt),
    "form": ("txt", _write_txt),
}

###############################################################################
# CLI helpers
###############################################################################


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic documents for HeronAI demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--count", type=int, default=1000, help="Total documents to generate."
    )
    parser.add_argument(
        "--out",
        dest="out_dir",
        type=Path,
        default=Path("datasets/synthetic"),
        help="Directory where documents will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    return parser.parse_args()


def _summarise(generated: Dict[str, int]) -> None:
    """Print a per-label summary table to stdout."""

    print("\n=== Generation summary ===")
    for label in _LABELS:
        print(f"{label:17} : {generated.get(label, 0)}")


###############################################################################
# Main entry-point
###############################################################################


def main() -> None:  # noqa: D401
    args = _parse_args()

    if args.count <= 0:
        print("--count must be a positive integer", file=sys.stderr)
        sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)
        Faker.seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    generated: Dict[str, int] = {label: 0 for label in _LABELS}

    for _ in range(args.count):
        label: str = random.choice(_LABELS)
        ext, writer = _WRITERS[label]
        text_content: str = _build_text(label)

        # Sub-directory per label keeps dataset tidy
        sub_dir: Path = args.out_dir / label
        sub_dir.mkdir(parents=True, exist_ok=True)

        filename: str = f"{label}_{uuid.uuid4().hex[:8]}.{ext}"
        path: Path = sub_dir / filename
        writer(path, text_content)
        generated[label] += 1

    _summarise(generated)
    total_size_mb: float = sum(p.stat().st_size for p in args.out_dir.rglob("*")) / (
        1024 * 1024
    )
    print(f"\nTotal size: {total_size_mb:.2f} MB written to {args.out_dir}")


if __name__ == "__main__":  # pragma: no cover – CLI only
    main()
