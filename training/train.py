"""Train a sentiment model and save vectorizer + model with joblib.
Usage: python train.py --data path/to/data.csv
Expect CSV with columns: text,label where label is 0/1 or neg/pos
"""
import argparse
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.preprocessing import preprocess_corpus
from app.sentiment_utils import normalize_binary_label

DEFAULT_ITEMS = [
    "page",
    "app",
    "product",
    "service",
    "design",
    "support",
    "interface",
    "experience",
    "tool",
    "website",
]

POSITIVE_EXTRAS = [
    "This is not bad at all",
    "I am happy with the results",
    "The workflow feels smooth and intuitive",
    "The support team was friendly and helpful",
    "Everything works exactly as expected",
    "The performance is fast and reliable",
    "This update made the experience better",
    "I would gladly recommend this to others",
]

NEGATIVE_EXTRAS = [
    "This is not good at all",
    "I am unhappy with the results",
    "The workflow feels slow and confusing",
    "The support team was unhelpful and rude",
    "Everything breaks when I use it",
    "The performance is slow and unreliable",
    "This update made the experience worse",
    "I would not recommend this to others",
]


def build_default_training_data():
    positive = []
    negative = []

    for item in DEFAULT_ITEMS:
        positive.extend(
            [
                f"I love this {item}",
                f"This {item} is amazing",
                f"The {item} is easy to use",
            ]
        )
        negative.extend(
            [
                f"I hate this {item}",
                f"This {item} is terrible",
                f"The {item} is hard to use",
            ]
        )

    positive.extend(POSITIVE_EXTRAS)
    negative.extend(NEGATIVE_EXTRAS)

    data = [(text, 1) for text in positive]
    data.extend((text, 0) for text in negative)
    return data


def load_data(path: str):
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have 'text' and 'label' columns")
    labels = [normalize_binary_label(label) for label in df["label"].tolist()]
    return df["text"].astype(str).tolist(), labels


def get_default_data():
    texts, labels = zip(*build_default_training_data())
    return list(texts), list(labels)


def should_create_holdout(labels) -> bool:
    label_counts = Counter(labels)
    return len(labels) >= 10 and min(label_counts.values()) >= 2


def build_metadata(
    args,
    labels,
    y_eval,
    predictions,
    report,
):
    counts = Counter(labels)
    return {
        "trained_at": datetime.now(UTC).isoformat(),
        "model_type": "LogisticRegression",
        "dataset": {
            "source": args.data if args.data else "built-in-default",
            "total_samples": len(labels),
            "class_distribution": {str(key): value for key, value in counts.items()},
        },
        "vectorizer": {
            "max_features": args.max_features,
            "ngram_range": [1, args.ngram_max],
            "sublinear_tf": True,
        },
        "evaluation": {
            "accuracy": round(accuracy_score(y_eval, predictions), 4),
            "macro_f1": round(f1_score(y_eval, predictions, average="macro"), 4),
            "classification_report": report,
        },
        "deployment": {
            "refit_on_full_dataset": True,
        },
    }


def build_vectorizer(args):
    return TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        sublinear_tf=True,
    )


def build_classifier():
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
    )


def main(args):
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    if args.data and os.path.exists(args.data):
        texts, labels = load_data(args.data)
    else:
        texts, labels = get_default_data()

    X = preprocess_corpus(texts)

    if should_create_holdout(labels):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )
    else:
        X_train, y_train = X, labels
        X_test, y_test = [], []
        print("Dataset is too small for a reliable holdout split; training on all data.")

    vect = build_vectorizer(args)
    X_train_t = vect.fit_transform(X_train)

    model = build_classifier()
    model.fit(X_train_t, y_train)

    if X_test:
        X_test_t = vect.transform(X_test)
        preds = model.predict(X_test_t)
        report = classification_report(y_test, preds, zero_division=0, output_dict=True)
        print(classification_report(y_test, preds, zero_division=0))
        y_eval = y_test
    else:
        train_preds = model.predict(X_train_t)
        report = classification_report(
            y_train,
            train_preds,
            zero_division=0,
            output_dict=True,
        )
        print(classification_report(y_train, train_preds, zero_division=0))
        preds = train_preds
        y_eval = y_train

    metadata = build_metadata(args, labels, y_eval, preds, report)

    final_vectorizer = build_vectorizer(args)
    X_full_t = final_vectorizer.fit_transform(X)
    final_model = build_classifier()
    final_model.fit(X_full_t, labels)

    joblib.dump(final_model, models_dir / "sentiment_model.pkl")
    joblib.dump(final_vectorizer, models_dir / "vectorizer.pkl")
    (models_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(f"Saved model and vectorizer to {models_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        help="Path to CSV with text,label columns",
        default="",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum number of TF-IDF features",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        choices=(1, 2, 3),
        help="Upper bound for the TF-IDF n-gram range",
    )
    args = parser.parse_args()
    main(args)
