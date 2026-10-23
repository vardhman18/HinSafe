
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

DEFAULT_MODEL_NAME = "google/muril-base-cased"
DEFAULT_MAX_LENGTH = 128
DEFAULT_NUM_LABELS = 2


def validate_data(dataframe):
    """Ensure that a dataframe contains the columns required for training."""

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing_columns)}"
        )


def load_tokenizer(model_name=DEFAULT_MODEL_NAME):
    """Load a MuRIL tokenizer from Hugging Face."""

    return AutoTokenizer.from_pretrained(model_name)


def prepare_dataset(
    dataframe,
    tokenizer,
    max_length=DEFAULT_MAX_LENGTH,
):
    """Convert and tokenize a dataframe for Hugging Face Trainer."""

    validate_data(dataframe)

    dataset = Dataset.from_pandas(
        dataframe[["text", "label"]],
        preserve_index=False,
    )

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    dataset = dataset.map(tokenize_batch, batched=True)

    columns_to_keep = {"input_ids", "attention_mask", "label"}
    columns_to_remove = [
        column
        for column in dataset.column_names
        if column not in columns_to_keep
    ]

    return dataset.remove_columns(columns_to_remove)


def create_model(
    model_name=DEFAULT_MODEL_NAME,
    num_labels=DEFAULT_NUM_LABELS,
):
    """Load MuRIL with a classification head."""

    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )


def compute_metrics(eval_prediction):
    """Calculate weighted classification metrics during evaluation."""

    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "precision": precision_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }


def train_model(
    train_df,
    val_df,
    model_name=DEFAULT_MODEL_NAME,
    output_dir="saved_models/muril",
    max_length=DEFAULT_MAX_LENGTH,
    batch_size=16,
    epochs=3,
    learning_rate=2e-5,
):
    """Fine-tune MuRIL and save the trained model and tokenizer."""

    tokenizer = load_tokenizer(model_name)
    train_dataset = prepare_dataset(
        train_df,
        tokenizer,
        max_length=max_length,
    )
    val_dataset = prepare_dataset(
        val_df,
        tokenizer,
        max_length=max_length,
    )
    model = create_model(model_name)

    output_path = Path(output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir=str(output_path / "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=42,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    return trainer, tokenizer


def predict_single(
    text,
    model,
    tokenizer,
    max_length=DEFAULT_MAX_LENGTH,
):
    """Return the predicted label and confidence for one text."""

    device = next(model.parameters()).device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits, dim=1)[0]

    label = torch.argmax(probabilities).item()
    confidence = probabilities[label].item()
    return label, confidence


def predict_texts(
    texts,
    model_path="saved_models/muril",
    max_length=DEFAULT_MAX_LENGTH,
):
    """Load a saved MuRIL model and predict labels for multiple texts."""

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory was not found: {model_path}"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits, dim=1)
        labels = torch.argmax(probabilities, dim=1)

    return labels.cpu().numpy(), probabilities.cpu().numpy()


def evaluate_model(
    trainer,
    test_df,
    tokenizer,
    max_length=DEFAULT_MAX_LENGTH,
):
    """Evaluate a trained MuRIL model on a labelled test dataframe."""

    test_dataset = prepare_dataset(
        test_df,
        tokenizer,
        max_length=max_length,
    )
    return trainer.evaluate(test_dataset)