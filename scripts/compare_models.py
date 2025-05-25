#!/usr/bin/env python3
"""
Model Comparison Tool for Document Classifier

This script compares the performance of the Naive Bayes and DistilBERT models
on a set of test documents to evaluate their classification accuracy and confidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
from faker import Faker
from tabulate import tabulate

# Initialize Faker for generating test data
faker = Faker()

# Test document templates
TEST_DOCUMENTS = {
    "invoice": [
        "Invoice #12345 for services rendered. Amount due: $500.00 USD. Payment due within 30 days.",
        "ACME Corp Tax Invoice. Order: XYZ789. Amount: $1,299.99. Please pay by bank transfer.",
        "Overdue Payment Notice: Your invoice #INV-567 from January is still unpaid. Please remit $750.00.",
    ],
    "bank_statement": [
        "Monthly Statement - Account #987654321. Beginning balance: $2,450.65. Ending balance: $3,102.45.",
        "BANK OF FINANCE - Quarterly Statement for checking account. Multiple transactions detected.",
        "Your account statement shows 5 deposits and 12 withdrawals. Current balance is $1,589.34.",
    ],
    "financial_report": [
        "Q2 Financial Performance Report. Revenue: $1.2M. EBITDA margin: 23%. Net profit increased by 5%.",
        "Annual financial analysis 2023. Balance sheet shows strong cash position and reduced debt ratio.",
        "Financial forecast for fiscal year 2024. Projected growth in key markets. ROI estimated at 15%.",
    ],
    "drivers_licence": [
        "Driver License #DL98765432. State of California. Class C. Expires: 05/15/2026.",
        "DMV Notice: Your driver's license renewal is due next month. Visit your local office to renew.",
        "Commercial driver license verification form. Vehicle class: Heavy vehicle (Class A).",
    ],
    "contract": [
        "Service Agreement between ACME Corp and XYZ Industries. Term: 24 months. Auto-renewal clause.",
        "Employment Contract. Position: Senior Developer. Start date: June 1. Compensation: $120,000.",
        "License Agreement for Software Use. The licensee hereby agrees to the terms set forth below.",
    ],
    "email": [
        "From: john.smith@company.com\nTo: support@vendor.com\nSubject: Question about order #12345\nI have an issue with my recent purchase.",
        "Email receipt for your online purchase. Order #XYZ789. Total amount: $89.99 including tax.",
        "Subject: Meeting invitation - Project kickoff\nDear team, I'd like to schedule a meeting next Tuesday at 10 AM.",
    ],
    "form": [
        "Application Form - Please complete all fields. Personal information: Name, Address, Contact details.",
        "Customer Feedback Survey. On a scale of 1-10, how would you rate our service? Additional comments:",
        "Health Insurance Enrollment Form. Coverage period: January 1 to December 31. Choose your plan below.",
    ],
}


def load_naive_bayes_model(model_path: Path) -> Any:
    """Load the Naive Bayes model."""
    if not model_path.exists():
        print(f"Error: Naive Bayes model not found at {model_path}")
        return None

    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        if (
            not isinstance(model_data, dict)
            or "vectoriser" not in model_data
            or "estimator" not in model_data
        ):
            print(f"Error: Invalid Naive Bayes model format at {model_path}")
            return None

        return model_data
    except Exception as e:
        print(f"Error loading Naive Bayes model: {str(e)}")
        return None


def predict_naive_bayes(
    model_data: Dict[str, Any], text: str
) -> Tuple[Optional[str], Optional[float]]:
    """Make a prediction using the Naive Bayes model."""
    if not text.strip() or model_data is None:
        return None, None

    vectoriser = model_data["vectoriser"]
    estimator = model_data["estimator"]

    try:
        X = vectoriser.transform([text])
        probas = estimator.predict_proba(X)[0]

        if hasattr(probas, "argmax"):
            predicted_index = int(probas.argmax())
        else:
            predicted_index = probas.index(max(probas))

        return estimator.classes_[predicted_index], float(probas[predicted_index])
    except Exception as e:
        print(f"Error in Naive Bayes prediction: {str(e)}")
        return None, None


def load_distilbert_module() -> Any:
    """Dynamically load the updated model.py module."""
    try:
        # Path to the model.py file
        model_file = Path("src/classification/model.py")

        if not model_file.exists():
            print(f"Error: model.py not found at {model_file}")
            return None

        # Create a module spec
        spec = importlib.util.spec_from_file_location("model_module", model_file)
        if spec is None or spec.loader is None:
            print("Error: Failed to create module spec")
            return None

        # Create the module
        module = importlib.util.module_from_spec(spec)
        sys.modules["model_module"] = module

        # Execute the module
        spec.loader.exec_module(module)

        # Return the module
        return module
    except Exception as e:
        print(f"Error loading DistilBERT module: {str(e)}")
        return None


def run_comparison(
    naive_bayes_path: Path, test_count: int = 3, verbose: bool = False
) -> None:
    """Run a comparison between the Naive Bayes and DistilBERT models."""
    # Load models
    naive_bayes_model = load_naive_bayes_model(naive_bayes_path)
    distilbert_module = load_distilbert_module()

    if naive_bayes_model is None and distilbert_module is None:
        print("Failed to load both models. Exiting.")
        return

    # Create test data
    all_results = []

    print("\nRunning model comparison...\n")

    # Process each document type
    for doc_type, templates in TEST_DOCUMENTS.items():
        for i, text in enumerate(templates[:test_count]):
            if verbose:
                print(f"\nTesting {doc_type} document #{i+1}:")
                print(f"Text: {text[:100]}...")

            # Add some random sentences for context
            text = text + " " + " ".join([faker.sentence() for _ in range(3)])

            result = {
                "type": doc_type,
                "text": text[:50] + "..." if len(text) > 50 else text,
            }

            # Test Naive Bayes
            if naive_bayes_model is not None:
                start_time = time.time()
                nb_label, nb_confidence = predict_naive_bayes(naive_bayes_model, text)
                nb_time = time.time() - start_time

                result["nb_label"] = nb_label
                result["nb_confidence"] = nb_confidence
                result["nb_correct"] = nb_label == doc_type
                result["nb_time_ms"] = nb_time * 1000

                if verbose:
                    print(
                        f"Naive Bayes: {nb_label} (confidence: {nb_confidence:.4f}, time: {nb_time*1000:.2f}ms)"
                    )
            else:
                result["nb_label"] = "N/A"
                result["nb_confidence"] = 0.0
                result["nb_correct"] = False
                result["nb_time_ms"] = 0.0

            # Test DistilBERT
            if distilbert_module is not None:
                try:
                    predict_func = getattr(distilbert_module, "predict")

                    start_time = time.time()
                    db_label, db_confidence = predict_func(text)
                    db_time = time.time() - start_time

                    result["db_label"] = db_label
                    result["db_confidence"] = db_confidence
                    result["db_correct"] = db_label == doc_type
                    result["db_time_ms"] = db_time * 1000

                    if verbose:
                        print(
                            f"DistilBERT: {db_label} (confidence: {db_confidence:.4f}, time: {db_time*1000:.2f}ms)"
                        )
                except Exception as e:
                    print(f"Error in DistilBERT prediction: {str(e)}")
                    result["db_label"] = "ERROR"
                    result["db_confidence"] = 0.0
                    result["db_correct"] = False
                    result["db_time_ms"] = 0.0
            else:
                result["db_label"] = "N/A"
                result["db_confidence"] = 0.0
                result["db_correct"] = False
                result["db_time_ms"] = 0.0

            all_results.append(result)

    # Calculate statistics
    if all_results:
        # Prepare table data
        table_data = []
        for r in all_results:
            table_data.append(
                [
                    r["type"],
                    r["text"],
                    r["nb_label"] if "nb_label" in r else "N/A",
                    f"{r['nb_confidence']:.4f}" if r.get("nb_confidence") else "N/A",
                    "✓" if r.get("nb_correct") else "✗",
                    r["db_label"] if "db_label" in r else "N/A",
                    f"{r['db_confidence']:.4f}" if r.get("db_confidence") else "N/A",
                    "✓" if r.get("db_correct") else "✗",
                ]
            )

        # Print results table
        print("\nResults:")
        print(
            tabulate(
                table_data,
                headers=[
                    "Document Type",
                    "Text",
                    "NB Prediction",
                    "NB Conf",
                    "NB",
                    "DB Prediction",
                    "DB Conf",
                    "DB",
                ],
                tablefmt="grid",
            )
        )

        # Calculate summary statistics
        nb_correct = sum(1 for r in all_results if r.get("nb_correct", False))
        db_correct = sum(1 for r in all_results if r.get("db_correct", False))
        total = len(all_results)

        nb_avg_conf = np.mean([r.get("nb_confidence", 0) for r in all_results])
        db_avg_conf = np.mean([r.get("db_confidence", 0) for r in all_results])

        nb_avg_time = np.mean([r.get("nb_time_ms", 0) for r in all_results])
        db_avg_time = np.mean([r.get("db_time_ms", 0) for r in all_results])

        # Print summary
        print("\nSummary:")
        print(
            f"Naive Bayes accuracy: {nb_correct}/{total} ({nb_correct/total*100:.1f}%)"
        )
        print(
            f"DistilBERT accuracy: {db_correct}/{total} ({db_correct/total*100:.1f}%)"
        )
        print(f"Naive Bayes avg confidence: {nb_avg_conf:.4f}")
        print(f"DistilBERT avg confidence: {db_avg_conf:.4f}")
        print(f"Naive Bayes avg processing time: {nb_avg_time:.2f}ms")
        print(f"DistilBERT avg processing time: {db_avg_time:.2f}ms")

        # Performance difference
        if nb_correct > 0 and db_correct > 0:
            acc_diff = (db_correct - nb_correct) / total * 100
            conf_diff = (db_avg_conf - nb_avg_conf) * 100
            time_ratio = db_avg_time / nb_avg_time if nb_avg_time > 0 else float("inf")

            print("\nPerformance difference:")
            print(
                f"Accuracy: {'DistilBERT' if acc_diff >= 0 else 'Naive Bayes'} is better by {abs(acc_diff):.1f}%"
            )
            print(
                f"Confidence: {'DistilBERT' if conf_diff >= 0 else 'Naive Bayes'} is higher by {abs(conf_diff):.1f}%"
            )
            print(
                f"Speed: DistilBERT is {time_ratio:.1f}x {'slower' if time_ratio > 1 else 'faster'} than Naive Bayes"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare document classification models"
    )
    parser.add_argument(
        "--naive-bayes",
        type=Path,
        default=Path("datasets/model.pkl"),
        help="Path to the Naive Bayes model pickle file",
    )
    parser.add_argument(
        "--test-count", type=int, default=3, help="Number of test documents per type"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress and results"
    )
    return parser.parse_args()


def main() -> None:
    try:
        # Check if tabulate is installed
        import tabulate
    except ImportError:
        print("Please install the tabulate package: pip install tabulate")
        return

    args = parse_args()
    run_comparison(
        naive_bayes_path=args.naive_bayes,
        test_count=args.test_count,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
