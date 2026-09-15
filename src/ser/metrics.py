"""Metrics and plots for SER experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from . import LABEL_NAMES


def classification_metrics(targets: list[int], predictions: list[int]) -> dict[str, object]:
    return {
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(targets, predictions, average="weighted", zero_division=0),
        "per_class": classification_report(
            targets, predictions, labels=list(range(len(LABEL_NAMES))), target_names=LABEL_NAMES,
            output_dict=True, zero_division=0,
        ),
    }


def save_evaluation(targets: list[int], predictions: list[int], output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = classification_metrics(targets, predictions)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(LABEL_NAMES))))
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=axis)
    axis.set(xlabel="Predicted", ylabel="True", title="Confusion matrix")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)
    return metrics

