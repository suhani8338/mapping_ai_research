"""Binary classification metrics for an explicitly selected positive label."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


def binary_metrics(
    truth: Iterable[T], predicted: Iterable[T], *, positive_label: T
) -> dict[str, float | int]:
    """Compute precision, recall, and false-positive rate; undefined rates are zero."""
    truth_values = list(truth)
    predicted_values = list(predicted)
    if len(truth_values) != len(predicted_values):
        raise ValueError("truth and predicted must have equal lengths")
    tp = fp = tn = fn = 0
    for actual, prediction in zip(truth_values, predicted_values):
        actual_positive = actual == positive_label
        predicted_positive = prediction == positive_label
        if actual_positive and predicted_positive:
            tp += 1
        elif predicted_positive:
            fp += 1
        elif actual_positive:
            fn += 1
        else:
            tn += 1
    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }
