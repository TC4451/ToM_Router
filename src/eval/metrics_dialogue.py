"""Dialogue-level evaluation metrics."""

import numpy as np
from src.eval.metrics_qa import exact_match, token_f1


def compute_adaptation_speed(turns: list[dict], transition_turn: int) -> dict:
    """Measure how quickly the policy adapts after a transition point.

    Returns turns-to-correct-route after the transition.
    """
    if transition_turn is None or transition_turn >= len(turns):
        return {"adaptation_speed": None, "post_transition_accuracy": None}

    post_turns = turns[transition_turn:]
    if not post_turns:
        return {"adaptation_speed": None, "post_transition_accuracy": None}

    # Find first correct route after transition
    first_correct = None
    correct_count = 0
    for i, t in enumerate(post_turns):
        if t.get("correct_route", False):
            if first_correct is None:
                first_correct = i
            correct_count += 1

    post_acc = correct_count / len(post_turns) if post_turns else 0

    return {
        "adaptation_speed": first_correct if first_correct is not None else len(post_turns),
        "post_transition_accuracy": post_acc,
    }


def compute_cost_quality_ratio(quality: float, tokens: int) -> float:
    """Compute quality per token (higher is better)."""
    if tokens == 0:
        return 0.0
    return quality / tokens * 1000  # quality per 1000 tokens


def compute_conversation_metrics(turn_results: list[dict], scenario: dict) -> dict:
    """Compute full metrics for one conversation under one policy."""
    n = len(turn_results)
    if n == 0:
        return {}

    # QA quality
    gold_answers = [t.get("gold_answer", "") for t in scenario["turns"][:n]]
    pred_answers = [t.get("response", "") for t in turn_results]
    em_scores = [exact_match(p, g) for p, g in zip(pred_answers, gold_answers)]
    f1_scores = [token_f1(p, g) for p, g in zip(pred_answers, gold_answers)]

    # Routing
    correct_routes = [t.get("correct_route", True) for t in turn_results]
    routes = [t.get("route", "unknown") for t in turn_results]

    # Cost
    total_tokens = sum(t.get("tokens_in", 0) + t.get("tokens_out", 0) for t in turn_results)
    total_latency = sum(t.get("latency_ms", 0) for t in turn_results)

    # Adaptation
    adapt = compute_adaptation_speed(turn_results, scenario.get("transition_turn"))

    avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0
    cqr = compute_cost_quality_ratio(avg_f1, total_tokens)

    return {
        "n_turns": n,
        "exact_match": float(np.mean(em_scores)),
        "token_f1": avg_f1,
        "routing_accuracy": float(np.mean(correct_routes)),
        "total_tokens": total_tokens,
        "avg_tokens_per_turn": total_tokens / n,
        "total_latency_ms": total_latency,
        "avg_latency_ms": total_latency / n,
        "n_tom_routes": sum(1 for r in routes if r == "tom"),
        "n_social_routes": sum(1 for r in routes if r == "social"),
        "cost_quality_ratio": cqr,
        **adapt,
    }
