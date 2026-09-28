from __future__ import annotations
import re
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "datasets" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "datasets" / "processed"

HATECOMMENTS_PATH = RAW_DATA_DIR / "hatecomments.csv"
CYBERBULLYING_PATH = RAW_DATA_DIR / "cyberbullying_dataset.csv"
HINGLISH_PATH = RAW_DATA_DIR / "hinglish.csv"

MERGED_FILE = PROCESSED_DATA_DIR / "hinsafe_final_dataset.csv"
TRAIN_FILE = PROCESSED_DATA_DIR / "train_split.csv"
VAL_FILE = PROCESSED_DATA_DIR / "val_split.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test_split.csv"


# stanaderdize the datasets to have the same format: text, label
def clean_hatecomments(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["text", "label"]].copy()
    df["label"] = df["label"].map({
        "offensive": 1,
        "not offensive": 0
    })
    return df


def clean_cyberbullying(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["Text", "Label"]].copy()
    df.columns = ["text", "label"]
    return df


def clean_hinglish(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["text", "hate_label"]].copy()
    df.columns = ["text", "label"]
    return df

#remove duplicates within each dataset and across datasets, clean text

def remove_duplicates(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    before = df.shape[0]
    df = df.drop_duplicates(subset="text", keep="first").copy()
    after = df.shape[0]
    removed = before - after

    print(f"{dataset_name}: removed {removed} duplicate rows ({before} -> {after})")
    return df


#clean text by lowercasing, removing URLs, mentions, hashtags, punctuation, and normalizing whitespace
def basic_clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_raw_datasets() -> Dict[str, pd.DataFrame]:
    return {
        "Hate Comments": pd.read_csv(HATECOMMENTS_PATH),
        "Cyberbullying": pd.read_csv(CYBERBULLYING_PATH),
        "Hinglish": pd.read_csv(HINGLISH_PATH),
    }


def standardize_datasets(raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    cleaned = {
        "Hate Comments": clean_hatecomments(raw_data["Hate Comments"]),
        "Cyberbullying": clean_cyberbullying(raw_data["Cyberbullying"]),
        "Hinglish": clean_hinglish(raw_data["Hinglish"]),
    }

    for dataset_name, df in cleaned.items():
        missing_labels = int(df["label"].isnull().sum())
        print(f"{dataset_name}: {missing_labels} rows with missing label after mapping")

    return cleaned



#merge the cleaned datasets, remove duplicates across datasets, clean text
def merge_and_clean_datasets(cleaned_datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    cleaned_frames = {}

    for dataset_name, df in cleaned_datasets.items():
        cleaned_frames[dataset_name] = remove_duplicates(df, dataset_name)

    merged_df = pd.concat(
        list(cleaned_frames.values()),
        ignore_index=True
    )

    before_cross_dataset = merged_df.shape[0]
    merged_df = merged_df.drop_duplicates(subset="text", keep="first").copy()
    after_cross_dataset = merged_df.shape[0]

    print(f"Removed {before_cross_dataset - after_cross_dataset} cross-dataset duplicate rows")
    print("Final merged shape:", merged_df.shape)

    merged_df = merged_df.dropna(subset=["text", "label"]).copy()
    merged_df["label"] = merged_df["label"].astype(int)

    print("Final shape after dropping empty rows:", merged_df.shape)
    print("\nFinal label distribution:")
    print(merged_df["label"].value_counts())
    print("\nFinal label distribution (percentage):")
    print((merged_df["label"].value_counts(normalize=True).round(3) * 100))

    merged_df["text"] = merged_df["text"].apply(basic_clean)
    merged_df = merged_df[merged_df["text"].str.strip() != ""].copy()
    merged_df = merged_df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    merged_df["label"] = merged_df["label"].astype(int)

    return merged_df


# split the merged dataset into train, validation, and test sets 

def split_dataset(merged_df: pd.DataFrame):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_val_df, test_df = train_test_split(
        merged_df,
        test_size=0.15,
        random_state=42,
        stratify=merged_df["label"]
    )

    val_ratio_of_remaining = 0.15 / 0.85

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_of_remaining,
        random_state=42,
        stratify=train_val_df["label"]
    )

    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    print("\nTrain label distribution:")
    print(train_df["label"].value_counts(normalize=True).round(3) * 100)

    print("\nValidation label distribution:")
    print(val_df["label"].value_counts(normalize=True).round(3) * 100)

    print("\nTest label distribution:")
    print(test_df["label"].value_counts(normalize=True).round(3) * 100)

    # Safety check: no sample should appear in more than one split
    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])

    overlap_train_val = train_texts.intersection(val_texts)
    overlap_train_test = train_texts.intersection(test_texts)
    overlap_val_test = val_texts.intersection(test_texts)

    print("\nOverlap between train and validation:", len(overlap_train_val))
    print("Overlap between train and test:", len(overlap_train_test))
    print("Overlap between validation and test:", len(overlap_val_test))

    if (
        len(overlap_train_val) == 0
        and len(overlap_train_test) == 0
        and len(overlap_val_test) == 0
    ):
        print("\nNo data leakage detected. Splits are clean.")
    else:
        print("\nWarning: overlapping text found between splits.")

    # Save the merged dataset and the split files
    merged_df.to_csv(MERGED_FILE, index=False)
    train_df.to_csv(TRAIN_FILE, index=False)
    val_df.to_csv(VAL_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print(f"\nSaved merged dataset to: {MERGED_FILE}")
    print(f"Saved train split to: {TRAIN_FILE}")
    print(f"Saved validation split to: {VAL_FILE}")
    print(f"Saved test split to: {TEST_FILE}")

    return train_df, val_df, test_df


def main() -> None:
    raw_data = load_raw_datasets()
    cleaned_datasets = standardize_datasets(raw_data)
    merged_df = merge_and_clean_datasets(cleaned_datasets)
    split_dataset(merged_df)


if __name__ == "__main__":
    main()