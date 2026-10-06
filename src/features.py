from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_features(
    train_texts,
    test_texts,
    ngram_range=(1, 2),
    max_features=20000
):
    """
    Create TF-IDF features for classical ML models.
    """

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
    
    """
    Tokenize and pad text for the BiLSTM model.
    """

    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

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