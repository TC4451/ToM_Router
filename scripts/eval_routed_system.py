"""Evaluate the full routed QA system against baselines.

Compares: single expert, oracle router, student router, always-ToM, always-social.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.seed import set_seed
from src.utils.config import load_config
from src.models.router_student import StudentRouter, get_tokenizer
from src.models.collators import RouterCollator
from src.training.trainer_distill import RouterDataset
from src.eval.metrics_router import compute_router_metrics


def load_student_router(model_path, model_name="microsoft/deberta-v3-base"):
    """Load trained student router."""
    model = StudentRouter(model_name=model_name)
    state_dict = torch.load(
        f"{model_path}/model.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict)
    return model


@torch.no_grad()
def get_router_probs(model, df, tokenizer, device, batch_size=32):
    """Get router probabilities for a dataframe."""
    dataset = RouterDataset(df.to_dict("records"))
    collator = RouterCollator(tokenizer, max_length=512)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=collator
    )

    model.eval()
    all_logits = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        all_logits.append(logits.cpu())

    logits = torch.cat(all_logits).numpy()
    return 1 / (1 + np.exp(-logits))


def evaluate_routing_strategy(
    labels: np.ndarray,
    route_decisions: np.ndarray,
    strategy_name: str,
) -> dict:
    """Evaluate a routing strategy."""
    route_accuracy = (route_decisions == labels).mean()
    tom_mask = labels == 1
    non_tom_mask = labels == 0

    return {
        "strategy": strategy_name,
        "route_accuracy": float(route_accuracy),
        "tom_recall": float(route_decisions[tom_mask].mean()) if tom_mask.sum() > 0 else 0.0,
        "non_tom_recall": float(1 - route_decisions[non_tom_mask].mean()) if non_tom_mask.sum() > 0 else 0.0,
        "n_routed_tom": int(route_decisions.sum()),
        "n_routed_social": int(len(route_decisions) - route_decisions.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_routed.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    # Load dataset
    dataset_path = config.get("dataset_path", "data/processed/router_dataset.parquet")
    df = pd.read_parquet(dataset_path)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    labels = test_df["requires_tom"].values

    print(f"Evaluating on test split: {len(test_df)} samples")

    # Load student router
    model_path = Path(config.get("router_path", "outputs/checkpoints/router_student")) / "best_f1"
    model_name = "microsoft/deberta-v3-base"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_student_router(model_path, model_name)
    model.to(device)
    tokenizer = get_tokenizer(model_name)

    # Get router predictions
    probs = get_router_probs(model, test_df, tokenizer, device)

    # Load best threshold from eval results
    eval_metrics_path = Path("outputs/reports/router_eval_metrics.json")
    threshold = 0.5
    if eval_metrics_path.exists():
        with open(eval_metrics_path) as f:
            eval_metrics = json.load(f)
            threshold = eval_metrics.get("thresholds", {}).get(
                "threshold_f1_best", {}
            ).get("threshold", 0.5)
    print(f"Using threshold: {threshold:.2f}")

    # Evaluate different routing strategies
    results = []

    # 1. Student router
    student_routes = (probs >= threshold).astype(int)
    results.append(evaluate_routing_strategy(labels, student_routes, "student_router"))
    student_metrics = compute_router_metrics(labels, probs, threshold)
    results[-1].update({
        "accuracy": student_metrics["accuracy"],
        "f1": student_metrics["f1"],
        "auroc": student_metrics["auroc"],
    })

    # 2. Oracle router (ground truth)
    oracle_routes = labels.copy()
    results.append(evaluate_routing_strategy(labels, oracle_routes, "oracle_router"))
    results[-1].update({"accuracy": 1.0, "f1": 1.0, "auroc": 1.0})

    # 3. Always ToM
    always_tom = np.ones_like(labels)
    results.append(evaluate_routing_strategy(labels, always_tom, "always_tom"))

    # 4. Always Social
    always_social = np.zeros_like(labels)
    results.append(evaluate_routing_strategy(labels, always_social, "always_social"))

    # 5. Random baseline
    rng = np.random.RandomState(42)
    random_routes = rng.randint(0, 2, size=len(labels))
    results.append(evaluate_routing_strategy(labels, random_routes, "random"))

    # Print comparison table
    print("\n" + "=" * 80)
    print("ROUTED SYSTEM COMPARISON")
    print("=" * 80)
    print(f"{'Strategy':<20} {'Route Acc':>10} {'ToM Recall':>12} {'Non-ToM Rec':>12} {'#ToM':>6} {'#Social':>8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['strategy']:<20} {r['route_accuracy']:>10.3f} "
            f"{r['tom_recall']:>12.3f} {r['non_tom_recall']:>12.3f} "
            f"{r['n_routed_tom']:>6} {r['n_routed_social']:>8}"
        )

    # Save results
    output_dir = Path(config.get("output_dir", "outputs/reports"))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "routed_system_eval.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save per-example routing decisions
    test_df = test_df.copy()
    test_df["student_prob_tom"] = probs
    test_df["student_route"] = np.where(probs >= threshold, "tom", "social")
    test_df["oracle_route"] = np.where(labels == 1, "tom", "social")
    test_df["route_correct"] = test_df["student_route"] == test_df["oracle_route"]
    test_df.to_parquet(output_dir / "routed_system_predictions.parquet", index=False)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
