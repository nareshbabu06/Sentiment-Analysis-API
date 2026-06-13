import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib

from .preprocessing import preprocess_text
from .sentiment_utils import format_sentiment_label, normalize_binary_label

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
SKLEARN_MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"
SKLEARN_VECT_PATH = MODELS_DIR / "vectorizer.pkl"
TRANSFORMER_MODEL_DIR = MODELS_DIR / "transformer_model"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


def _scalar(value):
    return value.item() if hasattr(value, "item") else value


class SentimentModel:
    def __init__(self):
        self.backend_name = "sklearn"
        self.model = None
        self.vectorizer = None
        self.tokenizer = None
        self.device = None
        self.metadata = {}

    def load(
        self,
        model_path: str = None,
        vect_path: str = None,
        metadata_path: str = None,
    ):
        metadata_file = Path(metadata_path or METADATA_PATH)
        self.metadata = self._load_metadata(metadata_file)
        self.backend_name = self.metadata.get("backend", "sklearn")

        if self.backend_name == "transformer":
            self._load_transformer(metadata_file)
            return

        self._load_sklearn(model_path=model_path, vect_path=vect_path)

    def is_loaded(self) -> bool:
        if self.backend_name == "transformer":
            return self.model is not None and self.tokenizer is not None
        return self.model is not None and self.vectorizer is not None

    def predict(self, text: str) -> Tuple[str, float]:
        details = self.predict_details(text)
        return details["raw_label"], details["confidence"]

    def predict_details(self, text: str) -> Dict[str, object]:
        if not self.is_loaded():
            raise RuntimeError("Sentiment model is not loaded.")
        if self.backend_name == "transformer":
            return self._predict_transformer(text)
        return self._predict_sklearn(text)

    def get_metadata(self) -> Dict[str, object]:
        if not self.metadata:
            return {"backend": self.backend_name}
        return self.metadata

    def _load_metadata(self, metadata_file: Path) -> Dict[str, object]:
        if metadata_file.exists():
            return json.loads(metadata_file.read_text())
        return {}

    def _load_sklearn(self, model_path: str = None, vect_path: str = None):
        model_file = Path(model_path or SKLEARN_MODEL_PATH)
        vectorizer_file = Path(vect_path or SKLEARN_VECT_PATH)
        self.model = joblib.load(model_file)
        self.vectorizer = joblib.load(vectorizer_file)
        self.tokenizer = None
        self.device = None

    def _load_transformer(self, metadata_file: Path):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Transformer dependencies are not installed. "
                "Install torch, transformers, datasets, and accelerate."
            ) from exc

        artifacts = self.metadata.get("artifacts", {})
        relative_model_dir = artifacts.get("model_dir", "models/transformer_model")
        model_dir = Path(relative_model_dir)
        if not model_dir.is_absolute():
            model_dir = Path(__file__).resolve().parents[1] / model_dir
        if not model_dir.exists():
            raise RuntimeError(
                f"Transformer model directory not found: {model_dir}"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.vectorizer = None

    def _predict_sklearn(self, text: str) -> Dict[str, object]:
        cleaned = preprocess_text(text)
        X = self.vectorizer.transform([cleaned])
        proba = self.model.predict_proba(X)[0]
        class_idx = proba.argmax()
        label = _scalar(self.model.classes_[class_idx])
        confidence = float(proba[class_idx])
        probability_map = {
            format_sentiment_label(_scalar(class_label)): float(probability)
            for class_label, probability in zip(self.model.classes_, proba)
        }
        return {
            "raw_label": label,
            "sentiment": format_sentiment_label(label),
            "confidence": confidence,
            "probabilities": probability_map,
            "top_terms": self._top_term_contributions(X, label),
        }

    def _predict_transformer(self, text: str) -> Dict[str, object]:
        import torch

        max_length = self.metadata.get("training", {}).get("max_length", 256)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0].cpu().tolist()

        class_idx = max(range(len(probabilities)), key=probabilities.__getitem__)
        raw_label = self.model.config.id2label.get(class_idx, class_idx)
        confidence = float(probabilities[class_idx])
        probability_map = {
            format_sentiment_label(self.model.config.id2label.get(index, index)): float(
                probability
            )
            for index, probability in enumerate(probabilities)
        }

        return {
            "raw_label": raw_label,
            "sentiment": format_sentiment_label(raw_label),
            "confidence": confidence,
            "probabilities": probability_map,
            "top_terms": [],
        }

    def _top_term_contributions(
        self,
        X,
        predicted_label,
        limit: int = 5,
    ) -> List[Dict[str, float]]:
        if not hasattr(self.model, "coef_"):
            return []

        feature_names = self.vectorizer.get_feature_names_out()
        row = X.tocoo()
        coefficients = self.model.coef_[0]
        predicted_value = normalize_binary_label(predicted_label)

        contributions = []
        for value, index in zip(row.data, row.col):
            contribution = float(coefficients[index] * value)
            score = contribution if predicted_value == 1 else -contribution
            contributions.append((feature_names[index], score))

        contributions.sort(key=lambda item: item[1], reverse=True)
        return [
            {"term": term, "contribution": round(score, 4)}
            for term, score in contributions[:limit]
            if score > 0
        ]


_SENTIMENT = SentimentModel()


def get_sentiment_model() -> SentimentModel:
    return _SENTIMENT
