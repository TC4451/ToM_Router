"""Evaluate the trained student router on the test split.

Generates metrics, confusion matrices, threshold analysis,
slice metrics, and error analysis report.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.seed import set_seed
from src.utils.config import load_config
from src.models.router_student import StudentRouter, get_tokenizer
from src.models.collators import RouterCollator
from src.training.trainer_distill import RouterDataset
from src.eval.metrics_router import (
    compute_router_metrics,
    compute_ece,
    compute_slice_metrics,
    find_multiple_thresholds,
)
from src.eval.error_analysis import generate_error_report


@torch.no_grad()
def get_predictions(model, dataloader, device) -> np.ndarray:
    """Run model on dataloader and return probabilities."""
    model.eval()
    all_logits = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        all_logits.append(logits.cpu())
    logits = torch.cat(all_logits).numpy()
    return 1 / (1 + np.exp(-logits))  # sigmoid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_router.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    # Load dataset
    dataset_path = config.get("dataset_path", "outputs/teacher_labels/router_dataset_with_teacher.parquet")
    if not Path(dataset_path).exists():
        dataset_path = "data/processed/router_dataset.parquet"
    df = pd.read_parquet(dataset_path)

    split = config.get("split", "test")
    test_df = df[df["split"] == split].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    print(f"Evaluating on {split} split: {len(test_df)} samples")

    # Load model
    model_path = Path(config.get("model_path", "outputs/checkpoints/router_student"))
    checkpoint_name = "best_f1"
    checkpoint_path = model_path / checkpoint_name

    model_name = "microsoft/deberta-v3-base"
    # Try to load config from checkpoint
    config_file = checkpoint_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            train_config = json.load(f)
            model_name = train_config.get("model_name", model_name)

    print(f"Loading model from {checkpoint_path}...")
    model = StudentRouter(model_name=model_name)
    model.load_state_dict(torch.load(checkpoint_path / "model.pt", map_location="cpu", weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = get_tokenizer(model_name)
    collator = RouterCollator(tokenizer, max_length=512)

    # Get predictions on test and val
    test_dataset = RouterDataset(test_df.to_dict("records"))
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collator)
    test_probs = get_predictions(model, test_loader, device)

    val_dataset = RouterDataset(val_df.to_dict("records"))
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collator)
    val_probs = get_predictions(model, val_loader, device)

    test_labels = test_df["requires_tom"].values
    val_labels = val_df["requires_tom"].values

    # Find best thresholds on validation
    print("\n--- Threshold Search (on validation) ---")
    thresholds = find_multiple_thresholds(val_labels, val_probs)
    for name, info in thresholds.items():
        print(f"  {name}: {info['threshold']:.2f} (val {name.split('_')[-1]}={info['value']:.4f})")

    best_threshold = thresholds["threshold_f1_best"]["threshold"]

    # Main metrics
    print(f"\n--- Test Metrics (threshold={best_threshold:.2f}) ---")
    metrics = compute_router_metrics(test_labels, test_probs, best_threshold)
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")

    # ECE
    ece = compute_ece(test_labels, test_probs)
    metrics["ece"] = ece
    print(f"  ece: {ece:.4f}")

    # Slice metrics by source
    print("\n--- Metrics by Source Dataset ---")
    source_metrics = compute_slice_metrics(
        test_labels, test_probs, test_df["source_dataset"].values, best_threshold
    )
    for source, m in source_metrics.items():
        print(f"  {source} (n={m['n_samples']}): acc={m['accuracy']:.3f}, f1={m['f1']:.3f}, auroc={m['auroc']:.3f}")

    # Slice metrics by subtype
    print("\n--- Metrics by Subtype ---")
    subtype_metrics = compute_slice_metrics(
        test_labels, test_probs, test_df["subtype"].values, best_threshold
    )
    for sub, m in subtype_metrics.items():
        print(f"  {sub} (n={m['n_samples']}): acc={m['accuracy']:.3f}, f1={m['f1']:.3f}")

    # Save results
    output_dir = Path(config.get("output_dir", "outputs/reports"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    all_results = {
        "main_metrics": metrics,
        "thresholds": thresholds,
        "source_metrics": source_metrics,
        "subtype_metrics": subtype_metrics,
    }
    with open(output_dir / "router_eval_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save per-example predictions
    if config.get("save_predictions", True):
        test_df = test_df.copy()
        test_df["student_prob_tom"] = test_probs
        test_df["student_pred"] = (test_probs >= best_threshold).astype(int)
        test_df.to_parquet(output_dir / "router_test_predictions.parquet", index=False)

    # Error analysis
    error_report = generate_error_report(test_df, test_probs, best_threshold)
    with open(output_dir / "router_error_analysis.txt", "w") as f:
        f.write(error_report)
    print(f"\nError analysis saved to {output_dir / 'router_error_analysis.txt'}")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
