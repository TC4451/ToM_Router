"""Run ablation studies for the ToM router.

Ablation groups:
1. Shortcut baselines (no neural model)
2. Distillation ablations (hard-only, soft-only, alpha/beta, temperature)
3. Dataset ablations (leave-one-source-out)
4. Student model ablation (DeBERTa vs DistilRoBERTa)
"""

import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.seed import set_seed
from src.models.router_student import StudentRouter, get_tokenizer
from src.models.collators import RouterCollator
from src.models.losses import DistillationLoss
from src.training.trainer_distill import RouterDataset, DistillationTrainer
from src.eval.metrics_router import compute_router_metrics

OUTPUT_DIR = Path("outputs/reports/ablations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = Path("outputs/teacher_labels/router_dataset_with_teacher.parquet")

# ============================================================
# SHORTCUT BASELINES
# ============================================================

def run_shortcut_baselines(df: pd.DataFrame) -> list[dict]:
    """Run shortcut baselines to detect dataset artifacts."""
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    y_train = train["requires_tom"].values
    y_test = test["requires_tom"].values
    results = []

    # 1. Majority class
    majority = int(pd.Series(y_train).mode().iloc[0])
    preds = np.full(len(y_test), majority)
    results.append({
        "ablation": "shortcut_majority_class",
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, average="macro")),
        "auroc": 0.5,
        "description": "Always predict majority class",
    })
    print(f"  Majority class: acc={results[-1]['accuracy']:.3f}, f1={results[-1]['f1']:.3f}")

    # 2. Source-only heuristic
    # Train a classifier using only source_dataset as feature
    source_map = {s: i for i, s in enumerate(df["source_dataset"].unique())}
    X_train_src = np.array([[source_map[s]] for s in train["source_dataset"]])
    X_test_src = np.array([[source_map[s]] for s in test["source_dataset"]])
    lr_src = LogisticRegression(random_state=42, max_iter=1000)
    lr_src.fit(X_train_src, y_train)
    preds_src = lr_src.predict(X_test_src)
    probs_src = lr_src.predict_proba(X_test_src)[:, 1]
    results.append({
        "ablation": "shortcut_source_only",
        "accuracy": float(accuracy_score(y_test, preds_src)),
        "f1": float(f1_score(y_test, preds_src, average="macro")),
        "auroc": float(roc_auc_score(y_test, probs_src)),
        "description": "Logistic regression on source_dataset feature only",
    })
    print(f"  Source-only: acc={results[-1]['accuracy']:.3f}, f1={results[-1]['f1']:.3f}, auroc={results[-1]['auroc']:.3f}")

    # 3. Context length only
    X_train_len = train["context"].str.len().values.reshape(-1, 1)
    X_test_len = test["context"].str.len().values.reshape(-1, 1)
    lr_len = LogisticRegression(random_state=42, max_iter=1000)
    lr_len.fit(X_train_len, y_train)
    preds_len = lr_len.predict(X_test_len)
    probs_len = lr_len.predict_proba(X_test_len)[:, 1]
    results.append({
        "ablation": "shortcut_length_only",
        "accuracy": float(accuracy_score(y_test, preds_len)),
        "f1": float(f1_score(y_test, preds_len, average="macro")),
        "auroc": float(roc_auc_score(y_test, probs_len)),
        "description": "Logistic regression on context length only",
    })
    print(f"  Length-only: acc={results[-1]['accuracy']:.3f}, f1={results[-1]['f1']:.3f}, auroc={results[-1]['auroc']:.3f}")

    # 4. Question keywords only
    q_vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
    X_train_q = q_vectorizer.fit_transform(train["question"])
    X_test_q = q_vectorizer.transform(test["question"])
    lr_q = LogisticRegression(random_state=42, max_iter=1000)
    lr_q.fit(X_train_q, y_train)
    preds_q = lr_q.predict(X_test_q)
    probs_q = lr_q.predict_proba(X_test_q)[:, 1]
    results.append({
        "ablation": "shortcut_question_keywords",
        "accuracy": float(accuracy_score(y_test, preds_q)),
        "f1": float(f1_score(y_test, preds_q, average="macro")),
        "auroc": float(roc_auc_score(y_test, probs_q)),
        "description": "Logistic regression on question TF-IDF (100 features)",
    })
    print(f"  Question keywords: acc={results[-1]['accuracy']:.3f}, f1={results[-1]['f1']:.3f}, auroc={results[-1]['auroc']:.3f}")

    # 5. Bag-of-words (context + question)
    bow_vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    texts_train = (train["context"] + " " + train["question"]).tolist()
    texts_test = (test["context"] + " " + test["question"]).tolist()
    X_train_bow = bow_vectorizer.fit_transform(texts_train)
    X_test_bow = bow_vectorizer.transform(texts_test)
    lr_bow = LogisticRegression(random_state=42, max_iter=1000)
    lr_bow.fit(X_train_bow, y_train)
    preds_bow = lr_bow.predict(X_test_bow)
    probs_bow = lr_bow.predict_proba(X_test_bow)[:, 1]
    results.append({
        "ablation": "shortcut_bow",
        "accuracy": float(accuracy_score(y_test, preds_bow)),
        "f1": float(f1_score(y_test, preds_bow, average="macro")),
        "auroc": float(roc_auc_score(y_test, probs_bow)),
        "description": "Logistic regression on TF-IDF bag-of-words (500 features)",
    })
    print(f"  Bag-of-words: acc={results[-1]['accuracy']:.3f}, f1={results[-1]['f1']:.3f}, auroc={results[-1]['auroc']:.3f}")

    return results


# ============================================================
# NEURAL MODEL TRAINING HELPER
# ============================================================

def train_and_eval(
    df: pd.DataFrame,
    config: dict,
    ablation_name: str,
    model_name: str = "microsoft/deberta-v3-base",
) -> dict:
    """Train a student router with given config and evaluate on test."""
    set_seed(config.get("seed", 42))

    train_records = df[df["split"] == "train"].to_dict("records")
    val_records = df[df["split"] == "val"].to_dict("records")
    test_records = df[df["split"] == "test"].to_dict("records")

    train_dataset = RouterDataset(train_records)
    val_dataset = RouterDataset(val_records)
    test_dataset = RouterDataset(test_records)

    model = StudentRouter(model_name=model_name)
    tokenizer = get_tokenizer(model_name)
    collator = RouterCollator(tokenizer, max_length=config.get("max_length", 512))

    output_dir = f"outputs/checkpoints/ablation_{ablation_name}"
    trainer = DistillationTrainer(
        model=model,
        collator=collator,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        output_dir=output_dir,
    )

    start = time.time()
    results = trainer.train()
    train_time = time.time() - start

    # Evaluate on test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load(f"{output_dir}/best_f1/model.pt", map_location="cpu", weights_only=True)
    )
    model.to(device)
    model.eval()

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, collate_fn=collator
    )

    all_logits = []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            all_logits.append(logits.cpu().float())

    logits = torch.cat(all_logits).numpy()
    probs = 1 / (1 + np.exp(-logits))
    labels = np.array([r["requires_tom"] for r in test_records])

    metrics = compute_router_metrics(labels, probs, threshold=0.5)

    # Free GPU
    del model
    torch.cuda.empty_cache()

    return {
        "ablation": ablation_name,
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "macro_f1": metrics["macro_f1"],
        "auroc": metrics["auroc"],
        "brier_score": metrics["brier_score"],
        "best_val_f1": results["best_f1"],
        "best_val_auroc": results["best_auroc"],
        "train_time_sec": train_time,
        "config": {k: v for k, v in config.items() if k != "seed"},
    }


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(42)
    all_results = []

    print("Loading dataset...")
    df = pd.read_parquet(DATASET_PATH)
    print(f"  {len(df)} samples")

    # ---- 1. Shortcut Baselines ----
    print("\n" + "=" * 60)
    print("1. SHORTCUT BASELINES")
    print("=" * 60)
    shortcut_results = run_shortcut_baselines(df)
    all_results.extend(shortcut_results)

    # ---- 2. Distillation Ablations ----
    print("\n" + "=" * 60)
    print("2. DISTILLATION ABLATIONS")
    print("=" * 60)

    base_config = {
        "seed": 42,
        "model_name": "microsoft/deberta-v3-base",
        "max_length": 512,
        "batch_size": 16,
        "lr": 2e-5,
        "weight_decay": 0.01,
        "epochs": 5,
        "warmup_ratio": 0.1,
        "scheduler": "cosine",
        "patience": 2,
    }

    distill_ablations = [
        # (name, alpha_hard, beta_soft, temperature, description)
        ("distill_hard_only", 1.0, 0.0, 1.5, "Hard labels only (no distillation)"),
        ("distill_soft_only", 0.0, 1.0, 1.5, "Soft labels only (no hard labels)"),
        ("distill_a07_b03_t15", 0.7, 0.3, 1.5, "Default: alpha=0.7, beta=0.3, T=1.5"),
        ("distill_a05_b05_t15", 0.5, 0.5, 1.5, "Equal weight: alpha=0.5, beta=0.5, T=1.5"),
        ("distill_a03_b07_t15", 0.3, 0.7, 1.5, "Soft-heavy: alpha=0.3, beta=0.7, T=1.5"),
        ("distill_a07_b03_t10", 0.7, 0.3, 1.0, "Low temp: alpha=0.7, beta=0.3, T=1.0"),
        ("distill_a07_b03_t20", 0.7, 0.3, 2.0, "High temp: alpha=0.7, beta=0.3, T=2.0"),
        ("distill_a07_b03_t30", 0.7, 0.3, 3.0, "Very high temp: alpha=0.7, beta=0.3, T=3.0"),
    ]

    for name, alpha, beta, temp, desc in distill_ablations:
        print(f"\n  Running: {desc}")
        config = deepcopy(base_config)
        config["alpha_hard"] = alpha
        config["beta_soft"] = beta
        config["distill_temperature"] = temp

        result = train_and_eval(df, config, name)
        result["description"] = desc
        all_results.append(result)
        print(f"    -> test acc={result['accuracy']:.4f}, f1={result['f1']:.4f}, "
              f"auroc={result['auroc']:.4f}")

    # ---- 3. Dataset Ablations (leave-one-out) ----
    print("\n" + "=" * 60)
    print("3. DATASET ABLATIONS (leave-one-source-out)")
    print("=" * 60)

    sources = df["source_dataset"].unique()
    for source in sources:
        n_source = len(df[df["source_dataset"] == source])
        if n_source < 10:
            continue  # skip tiny sources

        ablated_df = df[df["source_dataset"] != source].copy()

        # Check we still have both labels
        if ablated_df["requires_tom"].nunique() < 2:
            print(f"  Skipping {source}: removing it leaves only one class")
            continue

        # Re-balance
        for split in ["train", "val", "test"]:
            split_df = ablated_df[ablated_df["split"] == split]
            pos = split_df[split_df["requires_tom"] == 1]
            neg = split_df[split_df["requires_tom"] == 0]
            target = min(len(pos), len(neg))
            if target == 0:
                continue
            pos = pos.sample(n=target, random_state=42) if len(pos) > target else pos
            neg = neg.sample(n=target, random_state=42) if len(neg) > target else neg
            ablated_df = ablated_df[
                ~((ablated_df["split"] == split) &
                  (~ablated_df.index.isin(pd.concat([pos, neg]).index)))
            ]

        print(f"\n  Without {source}: {len(ablated_df)} samples "
              f"(removed {n_source})")

        config = deepcopy(base_config)
        config["alpha_hard"] = 0.7
        config["beta_soft"] = 0.3
        config["distill_temperature"] = 1.5

        name = f"dataset_without_{source}"
        result = train_and_eval(ablated_df, config, name)
        result["description"] = f"Without {source} ({n_source} removed)"
        result["removed_source"] = source
        result["removed_count"] = n_source
        result["remaining_samples"] = len(ablated_df)
        all_results.append(result)
        print(f"    -> test acc={result['accuracy']:.4f}, f1={result['f1']:.4f}, "
              f"auroc={result['auroc']:.4f}")

    # ---- 4. Student Model Ablation ----
    print("\n" + "=" * 60)
    print("4. STUDENT MODEL ABLATION")
    print("=" * 60)

    model_ablations = [
        ("model_distilroberta", "distilroberta-base", "DistilRoBERTa-base (82M params)"),
    ]

    for name, model_name, desc in model_ablations:
        print(f"\n  Running: {desc}")
        config = deepcopy(base_config)
        config["model_name"] = model_name
        config["alpha_hard"] = 0.7
        config["beta_soft"] = 0.3
        config["distill_temperature"] = 1.5

        result = train_and_eval(df, config, name, model_name=model_name)
        result["description"] = desc
        all_results.append(result)
        print(f"    -> test acc={result['accuracy']:.4f}, f1={result['f1']:.4f}, "
              f"auroc={result['auroc']:.4f}")

    # ---- Save All Results ----
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)

    # Print table
    print(f"\n{'Ablation':<40} {'Acc':>8} {'F1':>8} {'AUROC':>8}")
    print("-" * 68)
    for r in all_results:
        auroc_str = f"{r['auroc']:.4f}" if isinstance(r.get('auroc'), float) else str(r.get('auroc', 'N/A'))
        print(f"{r['ablation']:<40} {r['accuracy']:>8.4f} {r['f1']:>8.4f} {auroc_str:>8}")

    # Save
    with open(OUTPUT_DIR / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save as markdown table
    lines = ["# Ablation Results\n"]
    lines.append("| Ablation | Description | Acc | F1 | AUROC |")
    lines.append("|----------|-------------|-----|----|----- |")
    for r in all_results:
        auroc = f"{r['auroc']:.4f}" if isinstance(r.get('auroc'), float) else str(r.get('auroc', '-'))
        lines.append(
            f"| {r['ablation']} | {r.get('description', '')} | "
            f"{r['accuracy']:.4f} | {r['f1']:.4f} | {auroc} |"
        )
    with open(OUTPUT_DIR / "ablation_results.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
