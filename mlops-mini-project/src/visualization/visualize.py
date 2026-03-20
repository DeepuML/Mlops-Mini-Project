# -*- coding: utf-8 -*-
"""Visualisation utilities for the tweet emotion classification project.

Provides helper functions to create exploratory and results-oriented plots
that can be saved to ``reports/figures/`` or displayed inline in notebooks.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_class_distribution(y, labels: dict = None, title: str = "Class Distribution"):
    """Plot a bar chart of the target class distribution.

    Parameters
    ----------
    y:
        Array-like of integer class labels.
    labels:
        Optional mapping from integer label to display name,
        e.g. ``{0: "sadness", 1: "happiness"}``.
    title:
        Chart title.
    """
    unique, counts = np.unique(y, return_counts=True)
    display_labels = (
        [labels.get(int(u), str(u)) for u in unique] if labels else [str(u) for u in unique]
    )

    fig, ax = plt.subplots()
    ax.bar(display_labels, counts, color=["#4c9be8", "#f4a261"])
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, display_labels=None, title: str = "Confusion Matrix"):
    """Plot the confusion matrix for model predictions.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    display_labels:
        List of class names for the axes, e.g. ``["sadness", "happiness"]``.
    title:
        Chart title.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=display_labels
    )
    fig, ax = plt.subplots()
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_metric_comparison(results: dict, metric: str = "accuracy", title: str = None):
    """Plot a grouped bar chart comparing a single metric across experiments.

    Parameters
    ----------
    results:
        Mapping of ``{experiment_name: metric_value}``.
    metric:
        Name of the metric being plotted (used for the y-axis label).
    title:
        Chart title (defaults to ``metric`` if not provided).
    """
    names = list(results.keys())
    values = [results[n] for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
    bars = ax.bar(names, values, color="#4c9be8")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.capitalize())
    ax.set_title(title or metric.capitalize())
    ax.bar_label(bars, fmt="%.4f", padding=3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig
