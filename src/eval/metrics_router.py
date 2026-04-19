"""Router evaluation metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)


def compute_router_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute comprehensive router classification metrics."""
    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "brier_score": float(brier_score_loss(labels, probs)),
        "threshold": threshold,
    }

    try:
        metrics["auroc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["auroc"] = 0.5

    try:
        metrics["auprc"] = float(average_precision_score(labels, probs))
    except ValueError:
        metrics["auprc"] = 0.5

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    metrics["confusion_matrix"] = cm.tolist()
    if cm.shape == (2, 2):
        metrics["true_negatives"] = int(cm[0, 0])
        metrics["false_positives"] = int(cm[0, 1])
        metrics["false_negatives"] = int(cm[1, 0])
        metrics["true_positives"] = int(cm[1, 1])

    return metrics


def compute_ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(labels)

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() / total * abs(bin_acc - bin_conf)

    return float(ece)


def compute_slice_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    groups: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute metrics per slice (group)."""
    results = {}
    for group in np.unique(groups):
        mask = groups == group
        if mask.sum() < 5:
            continue
        results[str(group)] = compute_router_metrics(
            labels[mask], probs[mask], threshold
        )
        results[str(group)]["n_samples"] = int(mask.sum())
    return results


def find_best_threshold(labels: np.ndarray, probs: np.ndarray, metric: str = "f1") -> dict:
    """Search for optimal threshold on validation data."""
    thresholds = np.arange(0.1, 0.95, 0.01)
    best = {"threshold": 0.5, "value": 0.0}

    for t in thresholds:
        preds = (probs >= t).astype(int)
        if metric == "f1":
            val = f1_score(labels, preds, average="macro")
        elif metric == "recall":
            val = recall_score(labels, preds)
        else:
            val = accuracy_score(labels, preds)

        if val > best["value"]:
            best = {"threshold": float(t), "value": float(val)}

    return best


def find_multiple_thresholds(labels: np.ndarray, probs: np.ndarray) -> dict:
    """Find thresholds for multiple objectives."""
    return {
        "threshold_f1_best": find_best_threshold(labels, probs, "f1"),
        "threshold_high_recall_tom": find_best_threshold(labels, probs, "recall"),
        "threshold_accuracy_best": find_best_threshold(labels, probs, "accuracy"),
    }
