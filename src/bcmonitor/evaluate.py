from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def evaluate_predictions(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(y_true, y_pred, output_dict=False),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str],
    title: str,
    output_path: str | Path | None = None,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()


def print_evaluation_summary(name: str, results: dict) -> None:
    print(f"{name} accuracy: {results['accuracy']:.6f}")
    print()
    print(results["classification_report"])
    print()
    print(results["confusion_matrix"])


def save_feature_importance_plot(
    feature_names: list[str],
    importances: np.ndarray,
    title: str,
    output_path: str | Path | None = None,
) -> None:
    order = np.argsort(importances)[::-1]
    sorted_features = np.array(feature_names)[order]
    sorted_importances = importances[order]

    plt.figure(figsize=(10, 5))
    plt.bar(sorted_features, sorted_importances)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title(title)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()