from pathlib import Path
import pickle

from sklearn.svm import LinearSVC


def create_model(
    C=1.0,
    max_iter=10000,
    random_state=42
):
    """Create a Linear Support Vector Machine model."""

    return LinearSVC(
        C=C,
        max_iter=max_iter,
        random_state=random_state
    )


def train_model(
    X_train,
    y_train,
    C=1.0,
    max_iter=10000,
    random_state=42
):
    """Train and return a Linear SVM model."""

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


def decision_scores(model, X):
    """Return SVM decision scores."""

    return model.decision_function(X)


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