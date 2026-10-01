from pathlib import Path
import pickle

from sklearn.naive_bayes import MultinomialNB


def create_model(alpha=0.5):
    """Create a Multinomial Naive Bayes model."""

    return MultinomialNB(alpha=alpha)


def train_model(X_train, y_train, alpha=0.5):
    """Train and return a Multinomial Naive Bayes model."""

    model = create_model(alpha=alpha)
    model.fit(X_train, y_train)

    return model


def predict_model(model, X):
    """Generate class predictions."""

    return model.predict(X)


def predict_probabilities(model, X):
    """Generate class probabilities."""

    return model.predict_proba(X)


def save_model(model, model_path):
    """Save the trained model as a pickle file."""

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("wb") as file:
        pickle.dump(model, file)


def load_model(model_path):
    """Load a trained model from a pickle file."""

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file was not found: {model_path}"
        )

    with model_path.open("rb") as file:
        return pickle.load(file)