# -*- coding: utf-8 -*-
"""Train a Logistic Regression classifier on the processed tweet dataset
and log the run (parameters, metrics, and model artefacts) to MLflow.

Usage (from the project root):
    python src/models/train_model.py <processed_data_path> <model_output_dir>

Example:
    python src/models/train_model.py \\
        data/processed/tweet_emotions_processed.csv \\
        models/
"""

import logging
import os
import pickle
from pathlib import Path

import click
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default training hyper-parameters
# ---------------------------------------------------------------------------

DEFAULT_MAX_FEATURES = 1000
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

# Remote MLflow / DagsHub tracking URI
MLFLOW_TRACKING_URI = (
    "https://dagshub.com/DeepuML/Mlops-Mini-Project.mlflow"
)


# ---------------------------------------------------------------------------
# Training routine
# ---------------------------------------------------------------------------

def train(
    data_path: str,
    model_output_dir: str,
    max_features: int = DEFAULT_MAX_FEATURES,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict:
    """Train a Logistic Regression model and log the run to MLflow.

    Parameters
    ----------
    data_path:
        Path to the processed CSV file (columns: ``content``, ``sentiment``).
    model_output_dir:
        Directory where the trained model and vectoriser are saved.
    max_features:
        Maximum vocabulary size for ``CountVectorizer``.
    test_size:
        Fraction of data reserved for the test split.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    dict
        Evaluation metrics: accuracy, precision, recall, f1_score.
    """
    output_dir = Path(model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    log.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)
    X_text = df["content"]
    y = df["sentiment"]

    # ------------------------------------------------------------------
    # Feature extraction — Bag-of-Words
    # ------------------------------------------------------------------
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(X_text)

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    log.info(
        "Split: %d train / %d test samples", X_train.shape[0], X_test.shape[0]
    )

    # ------------------------------------------------------------------
    # MLflow experiment tracking
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Tweet Emotion Classification — Training")

    with mlflow.start_run():
        # Log hyper-parameters
        mlflow.log_param("vectorizer", "CountVectorizer")
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("model", "LogisticRegression")

        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="binary"),
            "recall": recall_score(y_test, y_pred, average="binary"),
            "f1_score": f1_score(y_test, y_pred, average="binary"),
        }
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        log.info(
            "Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f",
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
        )

        # Persist model and vectoriser
        model_path = output_dir / "logistic_regression.pkl"
        vectorizer_path = output_dir / "count_vectorizer.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)

        mlflow.log_artifact(str(model_path), "model")
        mlflow.log_artifact(str(vectorizer_path), "model")
        log.info("Model saved to %s", model_path)

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.argument("processed_data_path", type=click.Path(exists=True))
@click.argument("model_output_dir", type=click.Path())
@click.option(
    "--max-features",
    default=DEFAULT_MAX_FEATURES,
    show_default=True,
    help="Maximum vocabulary size for CountVectorizer.",
)
@click.option(
    "--test-size",
    default=DEFAULT_TEST_SIZE,
    show_default=True,
    help="Fraction of data used for evaluation.",
)
def main(
    processed_data_path: str,
    model_output_dir: str,
    max_features: int,
    test_size: float,
) -> None:
    """Train the tweet emotion classifier and log the run to MLflow."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )
    train(processed_data_path, model_output_dir, max_features, test_size)


if __name__ == "__main__":
    main()
