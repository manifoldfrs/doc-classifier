from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Tuple

from faker import Faker  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB  # type: ignore

faker: Faker = Faker()


_LABELS_AND_TEMPLATES: dict[str, List[str]] = {
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
        "Q{q} performance report",  # noqa: WPS323 – template placeholder
        "Balance sheet summary",
    ],
    "drivers_licence": [
        "Driver licence – {state} – ID {num}",
        "DL #{num} issued to {name}",
        "Driving license document",
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


def _render_template(template: str) -> str:  # noqa: D401
    """Replace placeholders in **template** with Faker-generated tokens."""

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


def _generate_samples(n: int) -> Tuple[List[str], List[str]]:  # noqa: D401
    """Return *(corpus, labels)* consisting of **n** synthetic documents."""

    corpus: List[str] = []
    labels: List[str] = []

    # Ensure roughly equal distribution across classes
    per_label: int = max(1, n // len(_LABELS_AND_TEMPLATES))
    for label, templates in _LABELS_AND_TEMPLATES.items():
        for _ in range(per_label):
            sentence = _render_template(faker.random_element(templates))
            # Faker might return unicode; lower-case for consistency
            corpus.append(sentence.lower())
            labels.append(label)

    return corpus, labels


def _train(samples: int) -> Tuple[TfidfVectorizer, MultinomialNB]:  # noqa: D401
    """Produce a fitted *(vectoriser, estimator)* pair."""

    X_raw, y = _generate_samples(samples)

    vectoriser = TfidfVectorizer(
        stop_words="english",
        max_features=5_000,
    )
    X = vectoriser.fit_transform(X_raw)

    clf = MultinomialNB()
    clf.fit(X, y)

    # Print quick evaluation for curiosity – not persisted
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\n=== Validation report ===")
    print(classification_report(y_test, y_pred, digits=3))

    return vectoriser, clf


def _parse_args() -> argparse.Namespace:  # noqa: D401
    parser = argparse.ArgumentParser(
        description="Train synthetic Naive Bayes model for demo"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3000,
        help="Number of synthetic samples to generate (default: 3000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/model.pkl"),
        help="Path where the pickle artefact will be written.",
    )
    return parser.parse_args()


def main() -> None:  # noqa: D401
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    vectoriser, estimator = _train(args.samples)

    with args.output.open("wb") as handle:
        pickle.dump({"vectoriser": vectoriser, "estimator": estimator}, handle)

    size_kb: float = args.output.stat().st_size / 1024
    print(f"\nModel saved ➜ {args.output} (size: {size_kb:.1f} KB)")


if __name__ == "__main__":  # pragma: no cover – CLI only
    main()
