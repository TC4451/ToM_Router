"""Extended ablation studies with visualizations.

Ablation groups:
1. Alpha/beta weight sweep (distillation weighting)
2. Temperature sweep
3. Training data size ablation
4. Contrastive ratio ablation
All run on hardened dataset with BERT-tiny (fast) + DeBERTa (key comparisons).
"""

import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.seed import set_seed
from src.models.router_student import StudentRouter, get_tokenizer
from src.models.collators import RouterCollator
from src.training.trainer_distill import RouterDataset, DistillationTrainer
from src.eval.metrics_router import compute_router_metrics

sns.set_theme(style="whitegrid", font_scale=1.1)
FIG_DIR = Path("outputs/figures")
OUT_DIR = Path("outputs/reports/ablations")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

HARD_PATH = Path("data/processed/router_dataset_hardened.parquet")
ORIG_PATH = Path("outputs/teacher_labels/router_dataset_with_teacher.parquet")


def train_and_eval(df, model_name, config, run_name):
    """Train one config, return test metrics."""
    set_seed(42)
    train_recs = df[df["split"] == "train"].to_dict("records")
    val_recs = df[df["split"] == "val"].to_dict("records")
    test_recs = df[df["split"] == "test"].to_dict("records")

    model = StudentRouter(model_name=model_name)
    n_params = sum(p.numel() for p in model.parameters())
    tokenizer = get_tokenizer(model_name)
    collator = RouterCollator(tokenizer, max_length=512)

    out_dir = f"outputs/checkpoints/abl_{run_name}"
    trainer = DistillationTrainer(
        model=model, collator=collator,
        train_dataset=RouterDataset(train_recs),
        val_dataset=RouterDataset(val_recs),
        config=config, output_dir=out_dir,
    )

    start = time.time()
    results = trainer.train()
    elapsed = time.time() - start

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load(f"{out_dir}/best_f1/model.pt", map_location="cpu", weights_only=True)
    )
    model.to(device).eval()

    loader = torch.utils.data.DataLoader(
        RouterDataset(test_recs), batch_size=32, collate_fn=collator
    )
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            all_logits.append(logits.cpu().float())

    logits = torch.cat(all_logits).numpy()
    probs = 1 / (1 + np.exp(-logits))
    labels = np.array([r["requires_tom"] for r in test_recs])
    metrics = compute_router_metrics(labels, probs, threshold=0.5)

    del model
    torch.cuda.empty_cache()

    return {
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "auroc": metrics["auroc"],
        "brier": metrics["brier_score"],
        "val_f1": results["best_f1"],
        "n_params": n_params,
        "train_sec": elapsed,
        "history": results["history"],
    }


def base_config():
    return {
        "seed": 42, "batch_size": 16, "lr": 2e-5, "weight_decay": 0.01,
        "epochs": 5, "warmup_ratio": 0.1, "scheduler": "cosine", "patience": 2,
        "alpha_hard": 0.7, "beta_soft": 0.3, "distill_temperature": 1.5,
    }


def ablation_alpha_beta(df):
    """Sweep alpha/beta weights."""
    print("\n=== ABLATION: Alpha/Beta Weight Sweep ===")
    configs = [
        (1.0, 0.0, "Hard only"),
        (0.9, 0.1, "90/10"),
        (0.7, 0.3, "70/30 (default)"),
        (0.5, 0.5, "50/50"),
        (0.3, 0.7, "30/70"),
        (0.1, 0.9, "10/90"),
        (0.0, 1.0, "Soft only"),
    ]
    results = []
    for alpha, beta, label in configs:
        print(f"  alpha={alpha}, beta={beta} ({label})...")
        cfg = base_config()
        cfg["alpha_hard"] = alpha
        cfg["beta_soft"] = beta
        r = train_and_eval(df, "google/bert_uncased_L-2_H-128_A-2", cfg, f"ab_{alpha}_{beta}")
        r["alpha"] = alpha
        r["beta"] = beta
        r["label"] = label
        results.append(r)
        print(f"    acc={r['accuracy']:.4f}, f1={r['f1']:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    alphas = [r["alpha"] for r in results]
    accs = [r["accuracy"] * 100 for r in results]
    f1s = [r["f1"] * 100 for r in results]
    ax.plot(alphas, accs, "o-", color="#E74C3C", label="Accuracy", linewidth=2, markersize=8)
    ax.plot(alphas, f1s, "s--", color="#3498DB", label="F1", linewidth=2, markersize=8)
    ax.set_xlabel("Alpha (hard label weight)")
    ax.set_ylabel("Score (%)")
    ax.set_title("Distillation Weight Sweep: Hard vs. Soft Label Balance\n(BERT-tiny on Hardened Dataset)")
    ax.legend()
    ax.set_xticks(alphas)
    ax.set_xticklabels([f"α={a:.1f}\nβ={1-a:.1f}" for a in alphas], fontsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    # Highlight default
    default_idx = next(i for i, r in enumerate(results) if r["alpha"] == 0.7)
    ax.axvline(x=0.7, color="gray", linestyle=":", alpha=0.5)
    ax.annotate("default", xy=(0.7, accs[default_idx]), xytext=(0.75, accs[default_idx]-1),
                fontsize=9, color="gray")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig12_alpha_beta_sweep.png", dpi=150)
    plt.close()
    print("  -> fig12_alpha_beta_sweep.png")
    return results


def ablation_temperature(df):
    """Sweep distillation temperature."""
    print("\n=== ABLATION: Temperature Sweep ===")
    temps = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    results = []
    for t in temps:
        print(f"  T={t}...")
        cfg = base_config()
        cfg["distill_temperature"] = t
        r = train_and_eval(df, "google/bert_uncased_L-2_H-128_A-2", cfg, f"temp_{t}")
        r["temperature"] = t
        results.append(r)
        print(f"    acc={r['accuracy']:.4f}, f1={r['f1']:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ts = [r["temperature"] for r in results]
    accs = [r["accuracy"] * 100 for r in results]
    briers = [r["brier"] * 100 for r in results]

    ax.plot(ts, accs, "o-", color="#E74C3C", label="Accuracy (%)", linewidth=2, markersize=8)
    ax.set_xlabel("Distillation Temperature (T)")
    ax.set_ylabel("Accuracy (%)", color="#E74C3C")
    ax.tick_params(axis="y", labelcolor="#E74C3C")

    ax2 = ax.twinx()
    ax2.plot(ts, briers, "s--", color="#9B59B6", label="Brier Score (×100)", linewidth=2, markersize=8)
    ax2.set_ylabel("Brier Score (×100, lower=better)", color="#9B59B6")
    ax2.tick_params(axis="y", labelcolor="#9B59B6")

    ax.set_title("Temperature Sweep: Effect on Accuracy and Calibration\n(BERT-tiny on Hardened Dataset)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower left")
    ax.axvline(x=1.5, color="gray", linestyle=":", alpha=0.5)
    for spine in ["top"]:
        ax.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig13_temperature_sweep.png", dpi=150)
    plt.close()
    print("  -> fig13_temperature_sweep.png")
    return results


def ablation_data_size(df):
    """Ablate training data size."""
    print("\n=== ABLATION: Training Data Size ===")
    fractions = [0.125, 0.25, 0.5, 0.75, 1.0]
    results = []
    for frac in fractions:
        train_df = df[df["split"] == "train"]
        if frac < 1.0:
            train_df = train_df.sample(frac=frac, random_state=42)
        rest = df[df["split"] != "train"]
        sub_df = pd.concat([train_df, rest])

        n_train = len(sub_df[sub_df["split"] == "train"])
        print(f"  frac={frac} ({n_train} train samples)...")

        cfg = base_config()
        r = train_and_eval(sub_df, "google/bert_uncased_L-2_H-128_A-2", cfg, f"size_{frac}")
        r["fraction"] = frac
        r["n_train"] = n_train
        results.append(r)
        print(f"    acc={r['accuracy']:.4f}, f1={r['f1']:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ns = [r["n_train"] for r in results]
    accs = [r["accuracy"] * 100 for r in results]
    f1s = [r["f1"] * 100 for r in results]
    ax.plot(ns, accs, "o-", color="#E74C3C", label="Accuracy", linewidth=2, markersize=8)
    ax.plot(ns, f1s, "s--", color="#3498DB", label="F1", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Training Samples")
    ax.set_ylabel("Score (%)")
    ax.set_title("Training Data Size Ablation\n(BERT-tiny, distilled, Hardened Dataset)")
    ax.legend()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig14_data_size_ablation.png", dpi=150)
    plt.close()
    print("  -> fig14_data_size_ablation.png")
    return results


def ablation_contrastive_ratio(orig_df, hard_df):
    """Ablate how much contrastive data is included."""
    print("\n=== ABLATION: Contrastive Ratio ===")
    from src.data.splits import balance_labels

    contrastive = hard_df[hard_df["source_dataset"].str.contains("contrastive", na=False)]
    original = hard_df[~hard_df["source_dataset"].str.contains("contrastive", na=False)]

    ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []

    for ratio in ratios:
        if ratio == 0.0:
            sub = original.copy()
        else:
            n_contrastive = int(len(contrastive) * ratio)
            sampled_c = contrastive.sample(n=n_contrastive, random_state=42)
            sub = pd.concat([original, sampled_c]).reset_index(drop=True)

        # Re-balance
        sub = balance_labels(sub, seed=42)
        n_total = len(sub)
        n_contr = len(sub[sub["source_dataset"].str.contains("contrastive", na=False)])

        print(f"  ratio={ratio} ({n_total} total, {n_contr} contrastive)...")

        cfg = base_config()
        r = train_and_eval(sub, "google/bert_uncased_L-2_H-128_A-2", cfg, f"contr_{ratio}")
        r["contrastive_ratio"] = ratio
        r["n_total"] = n_total
        r["n_contrastive"] = n_contr
        results.append(r)
        print(f"    acc={r['accuracy']:.4f}, f1={r['f1']:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ratios_plot = [r["contrastive_ratio"] * 100 for r in results]
    accs = [r["accuracy"] * 100 for r in results]
    ax.plot(ratios_plot, accs, "o-", color="#2ECC71", linewidth=2, markersize=10)
    ax.fill_between(ratios_plot, [min(accs) - 1] * len(ratios_plot), accs, alpha=0.1, color="#2ECC71")
    ax.set_xlabel("Contrastive Pairs Included (%)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Effect of Contrastive Augmentation Volume\n(BERT-tiny, distilled, Hardened Dataset)")
    for i, r in enumerate(results):
        ax.annotate(f"{r['n_contrastive']} pairs", xy=(ratios_plot[i], accs[i]),
                   xytext=(0, 10), textcoords="offset points", fontsize=8, ha="center")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig15_contrastive_ratio.png", dpi=150)
    plt.close()
    print("  -> fig15_contrastive_ratio.png")
    return results


def ablation_training_curves(df):
    """Show training curves for hard-only vs distilled."""
    print("\n=== ABLATION: Training Curves ===")

    cfg_hard = base_config()
    cfg_hard["alpha_hard"] = 1.0
    cfg_hard["beta_soft"] = 0.0
    r_hard = train_and_eval(df, "google/bert_uncased_L-2_H-128_A-2", cfg_hard, "curve_hard")

    cfg_dist = base_config()
    r_dist = train_and_eval(df, "google/bert_uncased_L-2_H-128_A-2", cfg_dist, "curve_dist")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curves
    ax = axes[0]
    epochs_h = [h["epoch"] + 1 for h in r_hard["history"]]
    epochs_d = [h["epoch"] + 1 for h in r_dist["history"]]
    ax.plot(epochs_h, [h["train_loss"] for h in r_hard["history"]], "o-", color="#E67E22", label="Hard only (train)")
    ax.plot(epochs_h, [h["val_loss"] for h in r_hard["history"]], "o--", color="#E67E22", alpha=0.6, label="Hard only (val)")
    ax.plot(epochs_d, [h["train_loss"] for h in r_dist["history"]], "s-", color="#9B59B6", label="Distilled (train)")
    ax.plot(epochs_d, [h["val_loss"] for h in r_dist["history"]], "s--", color="#9B59B6", alpha=0.6, label="Distilled (val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(fontsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # F1 curves
    ax = axes[1]
    ax.plot(epochs_h, [h["val_f1"] for h in r_hard["history"]], "o-", color="#E67E22", label="Hard only", linewidth=2)
    ax.plot(epochs_d, [h["val_f1"] for h in r_dist["history"]], "s-", color="#9B59B6", label="Distilled", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation F1")
    ax.set_title("Validation F1 Score")
    ax.legend()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.suptitle("Training Dynamics: Hard Labels Only vs. Distilled (BERT-tiny, Hardened Dataset)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig16_training_curves.png", dpi=150)
    plt.close()
    print("  -> fig16_training_curves.png")
    return r_hard, r_dist


def main():
    print("Loading datasets...")
    hard_df = pd.read_parquet(HARD_PATH)
    hard_df["teacher_prob_tom"] = hard_df["teacher_prob_tom"].fillna(
        hard_df["requires_tom"].astype(float)
    )
    orig_df = pd.read_parquet(ORIG_PATH)
    print(f"  Hardened: {len(hard_df)}, Original: {len(orig_df)}")

    all_results = {}

    all_results["alpha_beta"] = ablation_alpha_beta(hard_df)
    all_results["temperature"] = ablation_temperature(hard_df)
    all_results["data_size"] = ablation_data_size(hard_df)
    all_results["contrastive_ratio"] = ablation_contrastive_ratio(orig_df, hard_df)
    all_results["training_curves"] = "see fig16"
    ablation_training_curves(hard_df)

    # Save all results
    # Convert non-serializable items
    serializable = {}
    for k, v in all_results.items():
        if isinstance(v, list):
            serializable[k] = [{kk: vv for kk, vv in r.items() if kk != "history"} for r in v]
        else:
            serializable[k] = v

    with open(OUT_DIR / "extended_ablation_results.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"\nAll results saved to {OUT_DIR}/extended_ablation_results.json")
    print(f"All figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
