from pathlib import Path
import pickle

from sklearn.linear_model import LogisticRegression


def create_model(
    C=1.0,
    max_iter=1000,
    random_state=42
):
    """Create a Logistic Regression model."""

    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state
    )


def train_model(
    X_train,
    y_train,
    C=1.0,
    max_iter=1000,
    random_state=42
):
    """Train and return a Logistic Regression model."""

    model = create_model(
        C=C,
        max_iter=max_iter,
        random_state=random_state
    )

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