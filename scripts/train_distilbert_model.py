#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from faker import Faker
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# Initialize Faker for generating synthetic data
faker: Faker = Faker()

# Define document type templates (same as in original train_model.py)
_LABELS_AND_TEMPLATES: dict[str, List[str]] = {
    "invoice": [
        "Invoice #{num} – amount due {amount} {currency}",
        "Tax invoice for order {num}",
        "Payment reminder: Invoice {num}",
        "Your invoice is ready for payment. Please pay {amount} {currency} by bank transfer.",
        "Overdue invoice notification: {amount} {currency} payment required.",
    ],
    "bank_statement": [
        "Bank statement for account {num}",
        "Statement period {month} – ending balance {amount} {currency}",
        "Transaction list – acct {num}",
        "Your monthly account statement shows a balance of {amount} {currency}",
        "Banking summary for period ending {month} {year}",
    ],
    "financial_report": [
        "Annual financial report {year}",
        "Q{q} performance report",
        "Balance sheet summary",
        "Quarterly earnings report shows revenue of {amount} million {currency}",
        "Financial analysis for fiscal year {year} with projected growth",
    ],
    "drivers_licence": [
        "Driver licence – {state} – ID {num}",
        "DL #{num} issued to {name}",
        "Driving license document",
        "State of {state} driver's license verification document",
        "Commercial driver license renewal form",
    ],
    "contract": [
        "Service contract between {company1} and {company2}",
        "Employment agreement – {company1}",
        "Contract addendum #{num}",
        "Legal agreement for services rendered by {company1} to {company2}",
        "Contract terms and conditions for business partnership",
    ],
    "email": [
        "From: {name}@example.com\nTo: support@example.com\nSubject: Order {num}",
        "Subject: Inquiry about invoice {num}",
        "From: hr@{company1}.com\nSubject: Offer letter",
        "Email correspondence regarding project timeline and deliverables",
        "Customer support ticket #{num} via email",
    ],
    "form": [
        "Application form – {company1}",
        "Registration form for event {name}",
        "Order form #{num}",
        "Healthcare enrollment form for {name}",
        "Customer feedback survey form from {company1}",
    ],
}


def _render_template(template: str) -> str:
    """Replace placeholders in template with Faker-generated tokens."""
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


def _generate_samples(n: int) -> Tuple[List[str], List[str]]:
    """Return (corpus, labels) consisting of n synthetic documents."""
    corpus: List[str] = []
    labels: List[str] = []

    # Ensure roughly equal distribution across classes
    per_label: int = max(5, n // len(_LABELS_AND_TEMPLATES))
    for label, templates in _LABELS_AND_TEMPLATES.items():
        for _ in range(per_label):
            # Add more context to make samples richer for transformer model
            num_sentences = np.random.randint(1, 5)
            sentences = []

            # First sentence from template
            sentences.append(_render_template(faker.random_element(templates)))

            # Add some context sentences
            for _ in range(num_sentences):
                sentences.append(faker.sentence())

            # Join sentences and convert to lowercase for consistency
            document = " ".join(sentences).lower()
            corpus.append(document)
            labels.append(label)

    return corpus, labels


class DocumentClassificationDataset(Dataset):
    """PyTorch Dataset for document classification."""

    def __init__(
        self,
        texts: List[str],
        labels: List[str],
        tokenizer: DistilBertTokenizer,
        label2id: Dict[str, int],
    ):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,  # Limit sequence length to improve training speed
            return_tensors="pt",
        )
        self.labels = [label2id[label] for label in labels]

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def _train_distilbert(
    samples: int,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
) -> None:
    """Train and save a DistilBERT model for document classification."""
    print(f"Generating {samples} synthetic document samples...")
    texts, labels = _generate_samples(samples)

    # Create label mappings
    unique_labels = sorted(set(labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}

    print(f"Found {len(unique_labels)} unique document classes: {unique_labels}")

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(
        f"Training on {len(train_texts)} samples, validating on {len(val_texts)} samples"
    )

    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id,
    )

    # Create datasets
    train_dataset = DocumentClassificationDataset(
        train_texts, train_labels, tokenizer, label2id
    )
    val_dataset = DocumentClassificationDataset(
        val_texts, val_labels, tokenizer, label2id
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,  # Only keep the best model checkpoint
        learning_rate=learning_rate,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train the model
    print("Training DistilBERT model...")
    trainer.train()

    # Evaluate model
    print("\nEvaluating model...")
    result = trainer.evaluate()
    print(f"Evaluation results: {result}")

    # Run prediction to get classification report
    print("\nGenerating classification report...")
    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    pred_labels = [id2label[pred] for pred in preds]

    print(classification_report(val_labels, pred_labels, digits=3))

    # Save model, tokenizer and config
    model_dir = output_dir / "distilbert_model"

    # Clear previous model if it exists
    if model_dir.exists():
        shutil.rmtree(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving model to {model_dir}")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Ensure id2label is saved in the config file
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)

        # Ensure id2label is in the config
        if "id2label" not in config:
            config["id2label"] = {str(i): label for i, label in id2label.items()}
            config["label2id"] = {label: str(i) for label, i in label2id.items()}

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

    # Calculate model size
    model_size_bytes = sum(
        f.stat().st_size for f in model_dir.glob("**/*") if f.is_file()
    )
    model_size_mb = model_size_bytes / (1024 * 1024)

    print(f"Model saved to {model_dir} (size: {model_size_mb:.2f} MB)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DistilBERT model for document classification"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate (default: 1000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets"),
        help="Directory where the model will be saved",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Ensure output directory exists
    args.output.mkdir(parents=True, exist_ok=True)

    # Train and save model
    _train_distilbert(
        samples=args.samples,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
