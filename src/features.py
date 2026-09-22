from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_features(
    train_texts,
    test_texts,
    ngram_range=(1, 2),
    max_features=20000
):
    """
    TF-IDF features.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    return X_train, X_test, vectorizer