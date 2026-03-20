# -*- coding: utf-8 -*-
"""Feature engineering utilities for the tweet emotion classification task.

Provides helper functions to transform normalised tweet text into numerical
feature matrices suitable for scikit-learn classifiers.
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_bow_features(X_train, X_test, max_features: int = 1000):
    """Build Bag-of-Words feature matrices from pre-processed tweet text.

    Parameters
    ----------
    X_train:
        Iterable of normalised training tweet strings.
    X_test:
        Iterable of normalised test tweet strings.
    max_features:
        Maximum vocabulary size (keeps the top *max_features* by frequency).

    Returns
    -------
    tuple
        ``(X_train_vec, X_test_vec, vectorizer)`` — sparse matrices and the
        fitted :class:`~sklearn.feature_extraction.text.CountVectorizer`.
    """
    vectorizer = CountVectorizer(max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer


def build_tfidf_features(X_train, X_test, max_features: int = 1000):
    """Build TF-IDF feature matrices from pre-processed tweet text.

    Parameters
    ----------
    X_train:
        Iterable of normalised training tweet strings.
    X_test:
        Iterable of normalised test tweet strings.
    max_features:
        Maximum vocabulary size (keeps the top *max_features* by TF-IDF score).

    Returns
    -------
    tuple
        ``(X_train_vec, X_test_vec, vectorizer)`` — sparse matrices and the
        fitted :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer
