# -*- coding: utf-8 -*-
"""Download the raw tweet emotions dataset and prepare it for modelling.

Usage (from the project root):
    python src/data/make_dataset.py <raw_data_dir> <processed_data_dir>

Example:
    python src/data/make_dataset.py data/raw data/processed
"""

import logging
import os
import re
import string
import sys
from pathlib import Path

import click
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Remote URL for the original tweet emotions CSV
DATASET_URL = (
    "https://raw.githubusercontent.com/campusx-official/"
    "jupyter-masterclass/main/tweet_emotions.csv"
)

# Binary classes used for the classification task
TARGET_CLASSES = ["happiness", "sadness"]

# Label mapping applied after filtering
LABEL_MAP = {"sadness": 0, "happiness": 1}

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text preprocessing helpers
# ---------------------------------------------------------------------------

def lower_case(text: str) -> str:
    """Convert all characters in *text* to lower case."""
    return " ".join(word.lower() for word in text.split())


def remove_stop_words(text: str) -> str:
    """Remove English stop words from *text*."""
    stop_words = set(stopwords.words("english"))
    return " ".join(
        word for word in str(text).split() if word not in stop_words
    )


def remove_numbers(text: str) -> str:
    """Strip numeric characters from *text*."""
    return "".join(char for char in text if not char.isdigit())


def remove_punctuation(text: str) -> str:
    """Remove punctuation marks from *text*."""
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_urls(text: str) -> str:
    """Remove HTTP/HTTPS URLs from *text*."""
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub("", text)


def lemmatize(text: str) -> str:
    """Lemmatize each token in *text* using the WordNet lemmatiser."""
    lemmatizer = WordNetLemmatizer()
    return " ".join(lemmatizer.lemmatize(word) for word in text.split())


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full text normalisation pipeline to the *content* column.

    Steps applied in order:
    1. Lower-casing
    2. Stop-word removal
    3. Number removal
    4. Punctuation removal
    5. URL removal
    6. Lemmatisation

    Parameters
    ----------
    df:
        DataFrame with a ``content`` column containing raw tweet text.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with ``content`` replaced by normalised text.
    """
    try:
        df["content"] = df["content"].apply(lower_case)
        df["content"] = df["content"].apply(remove_stop_words)
        df["content"] = df["content"].apply(remove_numbers)
        df["content"] = df["content"].apply(remove_punctuation)
        df["content"] = df["content"].apply(remove_urls)
        df["content"] = df["content"].apply(lemmatize)
        return df
    except Exception as exc:
        log.error("Text normalisation failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.argument("raw_data_dir", type=click.Path())
@click.argument("processed_data_dir", type=click.Path())
def main(raw_data_dir: str, processed_data_dir: str) -> None:
    """Download the tweet emotions dataset, normalise it, and write the
    processed binary-class CSV to *processed_data_dir*.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    raw_dir = Path(raw_data_dir)
    processed_dir = Path(processed_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Download raw data
    # ------------------------------------------------------------------
    raw_path = raw_dir / "tweet_emotions.csv"
    log.info("Downloading dataset from %s", DATASET_URL)
    df = pd.read_csv(DATASET_URL).drop(columns=["tweet_id"])
    df.to_csv(raw_path, index=False)
    log.info("Raw data saved to %s  (%d rows)", raw_path, len(df))

    # ------------------------------------------------------------------
    # 2. Filter to binary classes
    # ------------------------------------------------------------------
    mask = df["sentiment"].isin(TARGET_CLASSES)
    df = df[mask].copy()
    log.info(
        "Filtered to %s classes: %d rows remaining",
        TARGET_CLASSES, len(df),
    )

    # ------------------------------------------------------------------
    # 3. Map labels to integers
    # ------------------------------------------------------------------
    df["sentiment"] = df["sentiment"].map(LABEL_MAP)

    # ------------------------------------------------------------------
    # 4. Normalise tweet text
    # ------------------------------------------------------------------
    log.info("Normalising tweet text …")
    df = normalize_text(df)

    # ------------------------------------------------------------------
    # 5. Save processed dataset
    # ------------------------------------------------------------------
    processed_path = processed_dir / "tweet_emotions_processed.csv"
    df.to_csv(processed_path, index=False)
    log.info(
        "Processed data saved to %s  (%d rows)", processed_path, len(df)
    )


if __name__ == "__main__":
    main()
