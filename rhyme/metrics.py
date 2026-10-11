"""Evaluation metrics for RHYME."""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from typing import Dict, Tuple


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 2,
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        predictions: Predicted labels
        targets: Ground truth labels
        num_classes: Number of classes

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, average="weighted", zero_division=0),
        "recall": recall_score(targets, predictions, average="weighted", zero_division=0),
        "f1": f1_score(targets, predictions, average="weighted", zero_division=0),
    }

    # Per-class metrics
    if num_classes == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(targets, predictions)
        except:
            metrics["roc_auc"] = 0.0

    return metrics


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(targets, predictions)


def print_metrics(metrics: Dict[str, float], task_name: str = "") -> None:
    """Pretty-print metrics."""
    if task_name:
        print(f"\n{task_name}")
        print("-" * 50)

    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name:15s}: {value:.4f}")
        else:
            print(f"{metric_name:15s}: {value}")


class MetricsTracker:
    """Track metrics during training/evaluation."""

    def __init__(self, num_tasks: int = 2):
        self.num_tasks = num_tasks
        self.reset()

    def reset(self) -> None:
        """Reset all metrics."""
        self.predictions = [[] for _ in range(self.num_tasks)]
        self.targets = [[] for _ in range(self.num_tasks)]

    def update(
        self,
        predictions: List[np.ndarray],
        targets: List[np.ndarray],
    ) -> None:
        """Update metrics with new predictions and targets."""
        for i in range(self.num_tasks):
            self.predictions[i].extend(predictions[i])
            self.targets[i].extend(targets[i])

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute and return all metrics."""
        results = {}
        task_names = ["deepfake", "generator"]

        for i, task_name in enumerate(task_names[:self.num_tasks]):
            if len(self.predictions[i]) > 0:
                results[task_name] = compute_metrics(
                    np.array(self.predictions[i]),
                    np.array(self.targets[i]),
                )

        return results
