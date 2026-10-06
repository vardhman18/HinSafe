from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf_features(
    train_texts,
    test_texts,
    ngram_range=(1, 2),
    max_features=20000
):
    """Create TF-IDF features for classical ML models."""

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    return X_train, X_test, vectorizer


def get_bilstm_features(
    train_texts,
    val_texts,
    test_texts,
    max_vocab_size=20000,
    max_sequence_length=50
):
    """Tokenize and pad text for the BiLSTM model."""

    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(
        num_words=max_vocab_size,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(train_texts)

    train_sequences = tokenizer.texts_to_sequences(train_texts)
    val_sequences = tokenizer.texts_to_sequences(val_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)

    X_train = pad_sequences(
        train_sequences,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )

    X_val = pad_sequences(
        val_sequences,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )

    X_test = pad_sequences(
        test_sequences,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )

    return X_train, X_val, X_test, tokenizer


def get_mbert_features(
    train_df,
    val_df,
    test_df,
    model_name="google-bert/bert-base-multilingual-cased",
    max_length=128
):
    """Tokenize train, validation, and test data for mBERT."""

    from datasets import Dataset
    from transformers import AutoTokenizer

    required_columns = {"text", "label"}

    for name, dataframe in (
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ):
        missing_columns = required_columns - set(dataframe.columns)

        if missing_columns:
            raise ValueError(
                f"{name} data is missing columns: {sorted(missing_columns)}"
            )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    datasets = [
        Dataset.from_pandas(
            dataframe[["text", "label"]],
            preserve_index=False
        )
        for dataframe in (train_df, val_df, test_df)
    ]

    tokenized_datasets = [
        dataset.map(tokenize_function, batched=True)
        for dataset in datasets
    ]

    columns_to_keep = [
        "input_ids",
        "attention_mask",
        "label"
    ]

    cleaned_datasets = []

    for dataset in tokenized_datasets:
        columns_to_remove = [
            column
            for column in dataset.column_names
            if column not in columns_to_keep
        ]

        cleaned_datasets.append(
            dataset.remove_columns(columns_to_remove)
        )

    return (*cleaned_datasets, tokenizer)