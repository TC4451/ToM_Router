"""QA evaluation metrics for the routed system."""

import numpy as np


def exact_match(prediction: str, reference: str) -> bool:
    """Check exact string match (case-insensitive, stripped)."""
    if not prediction or not reference:
        return False
    return prediction.strip().lower() == reference.strip().lower()


def token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between prediction and reference."""
    if not prediction or not reference:
        return 0.0

    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = pred_tokens & ref_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_qa_metrics(predictions: list[str], references: list[str]) -> dict:
    """Compute QA metrics over a list of predictions."""
    em_scores = [exact_match(p, r) for p, r in zip(predictions, references)]
    f1_scores = [token_f1(p, r) for p, r in zip(predictions, references)]

    return {
        "exact_match": float(np.mean(em_scores)),
        "token_f1": float(np.mean(f1_scores)),
        "n_samples": len(predictions),
    }
