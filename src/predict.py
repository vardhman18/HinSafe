
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from src.preprocessing import basic_clean
else:
    from preprocessing import basic_clean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "saved_models" / "muril"
DEFAULT_MAX_LENGTH = 128
TOP_N_WORDS = 5
OFFENSIVE_LABEL = "OFFENSIVE"
OFFENSIVE_THRESHOLD = 0.5

CATEGORY_LEXICON: dict[str, list[str]] = {
    "cyberbullying": [
        "bekaar", "ghatiya", "nalayak", "pagal", "bewakoof",
        "harass", "troll", "insult", "loser", "sadiyal", "suar",
        "idiot", "stupid", "moron", "useless", "nikamma", "nalla",
        "gawar", "jahil", "chapri", "chhapri", "aukat",
    ],
    "direct threat": [
        "mar ja", "maar dunga", "maarunga", "marunga",
        "khatam kar dunga", "jaan se maar", "jaan se maarunga",
        "dekh lunga", "dekh loonga", "tod dunga", "jala dunga",
        "uda dunga", "goli maar", "zinda nahi chhodunga",
        "kill", "murder", "rape", "dhamki", "threat",
    ],
    "misogyny": [
        "woman", "women", "girl", "ladki", "larki", "aurat",
        "bitch", "randi", "slut", "whore", "chhinal", "chinal",
        "characterless", "item", "maal",
    ],
    "communal / religious hate": [
        "hindu", "muslim", "musalman", "islam", "mandir", "masjid",
        "kaafir", "kafir", "religion", "katua", "mulla", "mulle",
        "jihadi", "aatankwadi", "terrorist", "paki",
    ],
    "casteist abuse": [
        "caste", "dalit", "chamar", "bhangi", "harijan", "neech",
        "neech jaat", "low caste", "chuhra", "achhoot", "untouchable",
    ],
    "homophobic / transphobic abuse": [
        "hijra", "chakka", "kinnar", "homo", "faggot", "tranny",
    ],
    "ableist abuse": [
        "retard", "retarded", "andha", "langda", "lula", "gunga",
        "behra",
    ],
    "appearance-based abuse": [
        "ugly", "mota", "moti", "fatso", "kala", "kali", "ganja",
        "takla", "suar jaisa",
    ],
    "obscene abuse": [
        "gaali", "gali", "chutiya", "madarchod", "maderchod",
        "behenchod", "bhenchod", "haramkhor", "kamina", "kutta",
        "fuck", "mother fuck", "motherfuck", "asshole", "bastard",
        "son of a bitch", "dumbass", "bullshit", "harami", "randwa",
    ],
}


def clean_text(text: str) -> str:
    """Normalize user input in the same way as the training pipeline."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return basic_clean(text)


def merge_into_words(
    tokens: Sequence[Any],
    values: Sequence[Any],
) -> tuple[list[str], list[float]]:
    """Merge SHAP subword tokens into whole words and sum their values."""

    words = []
    scores = []
    current_word = ""
    current_score = 0.0

    for token, value in zip(tokens, values):
        token = str(token)
        value = float(value)

        if not token.strip():
            continue

        if token[0].isspace() and current_word:
            words.append(current_word)
            scores.append(current_score)
            current_word = ""
            current_score = 0.0

        current_word += token.strip()
        current_score += value

        if token.endswith(" "):
            words.append(current_word)
            scores.append(current_score)
            current_word = ""
            current_score = 0.0

    if current_word:
        words.append(current_word)
        scores.append(current_score)

    return words, scores


def get_flagged_words(
    explanation: Any,
    top_n: int = TOP_N_WORDS,
) -> dict[str, float]:
    """Return the strongest words contributing toward OFFENSIVE."""

    words, scores = merge_into_words(explanation.data, explanation.values)
    positive_pairs = [
        (word, score)
        for word, score in zip(words, scores)
        if score > 0
    ]
    positive_pairs.sort(key=lambda pair: pair[1], reverse=True)
    return {
        word: round(score, 3)
        for word, score in positive_pairs[:top_n]
    }


def calculate_severity(offensive_probability: float) -> int:
    """Convert model confidence from 0-1 into a 0-100 severity score."""

    return int(round(float(offensive_probability) * 100))


def severity_band(score: int) -> str:
    """Map a 0-100 severity score to the public HinSafe band."""

    if score < 40:
        return "SAFE"
    if score < 70:
        return "OFFENSIVE"
    return "HATE SPEECH"


def detect_categories(
    clean_text: str,
    lexicon: dict[str, Sequence[str]] = CATEGORY_LEXICON,
) -> list[str]:
    """Return categories matched by complete words or phrases."""

    if not isinstance(clean_text, str):
        raise TypeError("clean_text must be a string")

    normalized_text = re.sub(r"\s+", " ", clean_text.lower()).strip()
    found_categories: list[str] = []

    for category, phrases in lexicon.items():
        for phrase in phrases:
            normalized_phrase = re.sub(r"\s+", " ", phrase.lower()).strip()
            if not normalized_phrase:
                continue

            pattern = rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)"
            if re.search(pattern, normalized_text):
                found_categories.append(category)
                break

    return found_categories


def get_offensive_probability(
    text: str,
    prediction_pipeline: Any,
) -> float:
    """Return the prediction pipeline's probability for the offensive class."""

    outputs = prediction_pipeline([text])[0]
    probabilities = {
        item["label"]: float(item["score"])
        for item in outputs
    }
    if OFFENSIVE_LABEL not in probabilities:
        raise KeyError(
            "OFFENSIVE label was not returned by the prediction pipeline."
        )
    return probabilities[OFFENSIVE_LABEL]


def analyze_comment(
    raw_text: str,
    prediction_pipeline: Any,
    explainer: Any = None,
    explain: bool = True,
) -> dict[str, Any]:
    """Analyze one Hinglish comment and optionally explain its result."""

    normalized_text = clean_text(raw_text)
    if not normalized_text:
        raise ValueError("Comment is empty after cleaning.")

    offensive_probability = get_offensive_probability(
        normalized_text,
        prediction_pipeline,
    )
    categories = detect_categories(normalized_text)
    label = int(
        offensive_probability >= OFFENSIVE_THRESHOLD
        or bool(categories)
    )
    if label:
        offensive_probability = max(
            offensive_probability,
            OFFENSIVE_THRESHOLD,
        )
    severity_score = calculate_severity(offensive_probability)

    flagged_words = {}
    if explain:
        if explainer is None:
            raise ValueError(
                "An explainer is required when explain=True."
            )
        shap_values = explainer([normalized_text])
        explanation = shap_values[0, :, OFFENSIVE_LABEL]
        flagged_words = get_flagged_words(explanation)

    return {
        "comment": raw_text,
        "label": label,
        "severity_score": severity_score,
        "severity_band": severity_band(severity_score),
        "categories": categories if label else [],
        "flagged_words": flagged_words,
    }


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
                "Install torch and transformers from requirements.txt."
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
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self._torch.no_grad():
            probabilities = self._torch.softmax(
                self.model(**inputs).logits,
                dim=1,
            )

        results: list[dict[str, Any]] = []
        for text, scores in zip(cleaned_texts, probabilities.cpu().tolist()):
            label = max(range(len(scores)), key=scores.__getitem__)
            offensive_probability = (
                scores[1] if len(scores) > 1 else scores[0]
            )
            categories = detect_categories(text)
            if categories:
                label = 1
                offensive_probability = max(
                    offensive_probability,
                    OFFENSIVE_THRESHOLD,
                )
            score = calculate_severity(offensive_probability)
            results.append(
                {
                    "comment": text,
                    "label": label,
                    "severity_score": score,
                    "severity_band": severity_band(score),
                    "categories": categories if label else [],
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
    comment = " ".join(args.text)
    print(
        json.dumps(
            predict_text(comment, args.model_path, args.max_length),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()