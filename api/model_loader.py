
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import shap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "saved_models" / "muril"
MODEL_PATH = os.getenv("HINSAFE_MODEL_PATH", str(DEFAULT_MODEL_PATH))
MAX_TOKEN_LENGTH = 128

ID_TO_LABEL = {
    0: "NOT_OFFENSIVE",
    1: "OFFENSIVE",
}
LABEL_TO_ID = {label: index for index, label in ID_TO_LABEL.items()}


def load_model_components() -> tuple[Any, Any]:
    """Load the prediction pipeline and SHAP explainer at API startup."""

    print(f"Loading MuRIL model from: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    model.eval()

    device = 0 if torch.cuda.is_available() else -1
    print(f"Running inference on: {'GPU' if device == 0 else 'CPU'}")

    prediction_pipeline = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
    )
    explainer = shap.Explainer(prediction_pipeline)

    return prediction_pipeline, explainer
