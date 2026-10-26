"""Inference utilities for the trained HinSafe MuRIL classifier."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "saved_models" / "muril"
DEFAULT_MAX_LENGTH = 128

_CATEGORY_KEYWORDS = {
    "cyberbullying": {"bully", "harass", "troll", "trolling", "insult", "loser"},
    "misogyny": {"woman", "women", "ladki", "larki", "girl", "bitch"},
    "communal / religious hate": {
        "hindu", "muslim", "islam", "religion", "mandir", "masjid",
    },
    "casteist abuse": {"caste", "dalit", "chamar", "bhangi"},
    "direct threat": {"kill", "maar", "mar", "murder", "destroy", "rape"},
}


def clean_text(text: str) -> str:
    """Normalize user input in the same way as the training pipeline."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _categories_for(text: str) -> list[str]:
    tokens = set(text.split())
    return [
        category
        for category, keywords in _CATEGORY_KEYWORDS.items()
        if tokens.intersection(keywords)
    ]


def severity_band(score: int) -> str:
    """Convert a 0-100 severity score into the public HinSafe band."""

    if score <= 40:
        return "SAFE"
    if score <= 70:
        return "OFFENSIVE"
    return "HATE SPEECH"


class Predictor:
    """Load a saved MuRIL model once and run predictions against it."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be greater than zero")

        model_path = Path(model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"Model directory was not found: {model_path}"
            )

        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as error:
            raise RuntimeError(
                "Prediction requires torch and transformers. "
                "Install dependencies from requirements.txt."
            ) from error

        self._torch = torch
        self.max_length = max_length
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path)
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict[str, Any]:
        """Return the HinSafe prediction schema for one comment."""

        return self.predict_many([text])[0]

    def predict_many(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        """Return prediction results for multiple comments in one batch."""

        if not texts:
            return []

        cleaned_texts = [clean_text(text) for text in texts]
        inputs = self.tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with self._torch.no_grad():
            probabilities = self._torch.softmax(
                self.model(**inputs).logits,
                dim=1,
            )

        results = []
        for text, scores in zip(cleaned_texts, probabilities.cpu().tolist()):
            label = max(range(len(scores)), key=scores.__getitem__)
            hate_probability = scores[1] if len(scores) > 1 else scores[0]
            score = round(hate_probability * 100)
            results.append(
                {
                    "comment": text,
                    "label": label,
                    "severity_score": score,
                    "severity_band": severity_band(score),
                    "categories": _categories_for(text) if label else [],
                    "flagged_words": {},
                }
            )

        return results


def predict_text(
    text: str,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict[str, Any]:
    """Load the saved model and predict one comment."""

    return Predictor(model_path, max_length).predict(text)


def predict_texts(
    texts: Sequence[str],
    model_path: str | Path = DEFAULT_MODEL_PATH,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> list[dict[str, Any]]:
    """Load the saved model and predict multiple comments."""

    return Predictor(model_path, max_length).predict_many(texts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict Hinglish hate speech with the HinSafe MuRIL model."
    )
    parser.add_argument("text", nargs="+", help="Comment to classify")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the saved MuRIL model directory",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum number of tokenizer tokens",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            predict_text(" ".join(args.text), args.model_path, args.max_length),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()