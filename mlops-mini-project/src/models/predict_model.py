# -*- coding: utf-8 -*-
"""Load a serialised model and vectoriser and generate predictions on new
tweet text.

Usage (from the project root):
    python src/models/predict_model.py <model_path> <vectorizer_path>

Example:
    python src/models/predict_model.py \\
        models/logistic_regression.pkl \\
        models/count_vectorizer.pkl
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import List

import click
import numpy as np

log = logging.getLogger(__name__)

# Label names corresponding to the integer targets used during training
LABEL_NAMES = {0: "sadness", 1: "happiness"}


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def load_artefacts(model_path: str, vectorizer_path: str):
    """Load and return the (model, vectorizer) pair from disk.

    Parameters
    ----------
    model_path:
        Path to the serialised scikit-learn model (`.pkl`).
    vectorizer_path:
        Path to the serialised ``CountVectorizer`` (`.pkl`).
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    log.info("Loaded model from %s", model_path)
    log.info("Loaded vectorizer from %s", vectorizer_path)
    return model, vectorizer


def predict(
    texts: List[str],
    model,
    vectorizer,
) -> List[str]:
    """Transform *texts* and return predicted emotion labels.

    Parameters
    ----------
    texts:
        List of raw (or pre-processed) tweet strings.
    model:
        Trained scikit-learn classifier.
    vectorizer:
        Fitted ``CountVectorizer`` (or equivalent) transformer.

    Returns
    -------
    list of str
        Predicted emotion label for each input tweet
        (``"happiness"`` or ``"sadness"``).
    """
    X = vectorizer.transform(texts)
    int_labels: np.ndarray = model.predict(X)
    return [LABEL_NAMES[label] for label in int_labels]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("vectorizer_path", type=click.Path(exists=True))
def main(model_path: str, vectorizer_path: str) -> None:
    """Interactively predict the emotion of tweets using a saved model.

    The command loads the model and vectoriser, then reads tweets from stdin
    (one per line) and prints the predicted label for each.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    model, vectorizer = load_artefacts(model_path, vectorizer_path)

    log.info("Enter tweets (one per line). Press Ctrl+D / Ctrl+Z to quit.")
    for line in sys.stdin:
        tweet = line.strip()
        if not tweet:
            continue
        label = predict([tweet], model, vectorizer)[0]
        print(f"Tweet : {tweet}")
        print(f"Prediction : {label}")
        print()


if __name__ == "__main__":
    main()
