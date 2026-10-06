from pathlib import Path
import pickle
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    Input,
    LSTM
)
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


DEFAULT_MAX_VOCAB_SIZE = 20000
DEFAULT_MAX_SEQUENCE_LENGTH = 50
DEFAULT_EMBEDDING_DIM = 300
DEFAULT_LSTM_UNITS = 128


def create_tokenizer(
    train_texts,
    max_vocab_size=DEFAULT_MAX_VOCAB_SIZE
):
    """Create and fit a tokenizer using training text only."""

    tokenizer = Tokenizer(
        num_words=max_vocab_size,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(train_texts)

    return tokenizer


def convert_texts_to_sequences(
    tokenizer,
    texts,
    max_sequence_length=DEFAULT_MAX_SEQUENCE_LENGTH
):
    """Convert text into padded integer sequences."""

    sequences = tokenizer.texts_to_sequences(texts)

    return pad_sequences(
        sequences,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )


def prepare_data(
    train_texts,
    val_texts,
    test_texts,
    max_vocab_size=DEFAULT_MAX_VOCAB_SIZE,
    max_sequence_length=DEFAULT_MAX_SEQUENCE_LENGTH
):
    """Tokenize and pad train, validation, and test text."""

    tokenizer = create_tokenizer(
        train_texts=train_texts,
        max_vocab_size=max_vocab_size
    )

    X_train = convert_texts_to_sequences(
        tokenizer=tokenizer,
        texts=train_texts,
        max_sequence_length=max_sequence_length
    )

    X_val = convert_texts_to_sequences(
        tokenizer=tokenizer,
        texts=val_texts,
        max_sequence_length=max_sequence_length
    )

    X_test = convert_texts_to_sequences(
        tokenizer=tokenizer,
        texts=test_texts,
        max_sequence_length=max_sequence_length
    )

    return X_train, X_val, X_test, tokenizer


def create_embedding_matrix(
    tokenizer,
    fasttext_model,
    vocab_size,
    embedding_dim=DEFAULT_EMBEDDING_DIM
):
    """Create an embedding matrix from a loaded FastText model."""

    embedding_matrix = np.zeros(
        (vocab_size, embedding_dim),
        dtype=np.float32
    )

    for word, index in tokenizer.word_index.items():
        if index >= vocab_size:
            continue

        embedding_matrix[index] = fasttext_model.get_word_vector(word)

    return embedding_matrix


def build_model(
    vocab_size,
    embedding_dim=DEFAULT_EMBEDDING_DIM,
    max_sequence_length=DEFAULT_MAX_SEQUENCE_LENGTH,
    lstm_units=DEFAULT_LSTM_UNITS,
    embedding_matrix=None,
    trainable_embeddings=False
):
    """Build and compile the BiLSTM classification model."""

    model = Sequential([
        Input(
            shape=(max_sequence_length,),
            dtype="int32"
        )
    ])

    if embedding_matrix is not None:
        model.add(
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                weights=[embedding_matrix],
                trainable=trainable_embeddings
            )
        )
    else:
        model.add(
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                trainable=trainable_embeddings
            )
        )

    model.add(Bidirectional(LSTM(lstm_units)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=10,
    batch_size=32,
    patience=3
):
    """Train the BiLSTM model and return the training history."""

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    return history


def predict_model(model, X):
    
    probabilities = model.predict(X, verbose=0).reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)

    return predictions, probabilities


def save_model(model, model_path):

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(str(model_path))


def load_trained_model(model_path):
    
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file was not found: {model_path}"
        )

    return keras_load_model(str(model_path))


def save_tokenizer(tokenizer, tokenizer_path):
    
    tokenizer_path = Path(tokenizer_path)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    with tokenizer_path.open("wb") as file:
        pickle.dump(tokenizer, file)


def load_tokenizer(tokenizer_path):
    
    tokenizer_path = Path(tokenizer_path)

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer file was not found: {tokenizer_path}"
        )

    with tokenizer_path.open("rb") as file:
        return pickle.load(file)