"""Run ablations on the hardened dataset to prove:
1. Source shortcut is broken
2. Distillation helps weak models more on harder data
"""

import json
import sys
import time
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
from src.training.trainer_distill import RouterDataset, DistillationTrainer
from src.eval.metrics_router import compute_router_metrics, compute_ece

OUTPUT_DIR = Path("outputs/reports/ablations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_PATH = Path("outputs/teacher_labels/router_dataset_with_teacher.parquet")
HARDENED_PATH = Path("data/processed/router_dataset_hardened.parquet")


def run_baselines(df, dataset_name):
    """Shortcut baselines."""
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    y_train = train["requires_tom"].values
    y_test = test["requires_tom"].values
    results = []

    # Majority
    majority = int(pd.Series(y_train).mode().iloc[0])
    preds = np.full(len(y_test), majority)
    results.append({
        "name": "Majority class",
        "dataset": dataset_name,
        "test_acc": float(accuracy_score(y_test, preds)),
        "test_f1": float(f1_score(y_test, preds, average="macro")),
        "test_auroc": 0.5,
    })

    # Source-only (use base source name, stripping "_contrastive")
    train_src = train["source_dataset"].str.replace("_contrastive", "", regex=False)
    test_src = test["source_dataset"].str.replace("_contrastive", "", regex=False)
    all_sources = list(set(train_src.unique()) | set(test_src.unique()))
    source_map = {s: i for i, s in enumerate(all_sources)}
    X_train = np.array([[source_map.get(s, 0)] for s in train_src])
    X_test = np.array([[source_map.get(s, 0)] for s in test_src])
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    probs = lr.predict_proba(X_test)[:, 1]
    preds = lr.predict(X_test)
    results.append({
        "name": "Source-only LR (base source)",
        "dataset": dataset_name,
        "test_acc": float(accuracy_score(y_test, preds)),
        "test_f1": float(f1_score(y_test, preds, average="macro")),
        "test_auroc": float(roc_auc_score(y_test, probs)),
    })

    # BoW
    texts_train = (train["context"] + " " + train["question"]).tolist()
    texts_test = (test["context"] + " " + test["question"]).tolist()
    vec = TfidfVectorizer(max_features=500, stop_words="english")
    X_train = vec.fit_transform(texts_train)
    X_test = vec.transform(texts_test)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    probs = lr.predict_proba(X_test)[:, 1]
    preds = lr.predict(X_test)
    results.append({
        "name": "TF-IDF BoW LR",
        "dataset": dataset_name,
        "test_acc": float(accuracy_score(y_test, preds)),
        "test_f1": float(f1_score(y_test, preds, average="macro")),
        "test_auroc": float(roc_auc_score(y_test, probs)),
    })

    return results


def train_and_eval(df, model_name, alpha, beta, temp, run_name, epochs=5):
    """Train and evaluate one config."""
    set_seed(42)
    train_recs = df[df["split"] == "train"].to_dict("records")
    val_recs = df[df["split"] == "val"].to_dict("records")
    test_recs = df[df["split"] == "test"].to_dict("records")

    model = StudentRouter(model_name=model_name)
    n_params = sum(p.numel() for p in model.parameters())
    tokenizer = get_tokenizer(model_name)
    collator = RouterCollator(tokenizer, max_length=512)

    config = {
        "seed": 42, "batch_size": 16, "lr": 2e-5, "weight_decay": 0.01,
        "epochs": epochs, "warmup_ratio": 0.1, "scheduler": "cosine",
        "patience": 2, "alpha_hard": alpha, "beta_soft": beta,
        "distill_temperature": temp,
    }

    out_dir = f"outputs/checkpoints/ablation_{run_name}"
    trainer = DistillationTrainer(
        model=model, collator=collator,
        train_dataset=RouterDataset(train_recs),
        val_dataset=RouterDataset(val_recs),
        config=config, output_dir=out_dir,
    )

    start = time.time()
    results = trainer.train()
    elapsed = time.time() - start

    # Evaluate on test
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

    del model; torch.cuda.empty_cache()

    return {
        "test_acc": metrics["accuracy"],
        "test_f1": metrics["f1"],
        "test_macro_f1": metrics["macro_f1"],
        "test_auroc": metrics["auroc"],
        "test_brier": metrics["brier_score"],
        "val_f1": results["best_f1"],
        "n_params": n_params,
        "train_sec": elapsed,
    }


def main():
    all_results = []

    # ---- ORIGINAL DATASET ----
    print("=" * 70)
    print("ORIGINAL DATASET (with source shortcuts)")
    print("=" * 70)
    orig_df = pd.read_parquet(ORIGINAL_PATH)
    print(f"  {len(orig_df)} samples\n")

    orig_baselines = run_baselines(orig_df, "original")
    all_results.extend(orig_baselines)
    for r in orig_baselines:
        print(f"  {r['name']:<30} acc={r['test_acc']:.4f}  f1={r['test_f1']:.4f}  auroc={r['test_auroc']:.4f}")

    # BERT-tiny hard vs distilled on original
    print("\n  BERT-tiny (4M) — hard only...")
    r = train_and_eval(orig_df, "google/bert_uncased_L-2_H-128_A-2", 1.0, 0.0, 1.5, "orig_tiny_hard")
    r.update({"name": "BERT-tiny hard", "dataset": "original", "distilled": "No"})
    all_results.append(r)
    print(f"    -> acc={r['test_acc']:.4f}  f1={r['test_f1']:.4f}  auroc={r['test_auroc']:.4f}")

    print("  BERT-tiny (4M) — distilled...")
    r = train_and_eval(orig_df, "google/bert_uncased_L-2_H-128_A-2", 0.7, 0.3, 1.5, "orig_tiny_dist")
    r.update({"name": "BERT-tiny distilled", "dataset": "original", "distilled": "Yes"})
    all_results.append(r)
    print(f"    -> acc={r['test_acc']:.4f}  f1={r['test_f1']:.4f}  auroc={r['test_auroc']:.4f}")

    # DeBERTa hard vs distilled on original
    print("\n  DeBERTa-v3-base — hard only...")
    r = train_and_eval(orig_df, "microsoft/deberta-v3-base", 1.0, 0.0, 1.5, "orig_deberta_hard")
    r.update({"name": "DeBERTa hard", "dataset": "original", "distilled": "No"})
    all_results.append(r)
    print(f"    -> acc={r['test_acc']:.4f}  f1={r['test_f1']:.4f}  auroc={r['test_auroc']:.4f}")

    print("  DeBERTa-v3-base — distilled...")
    r = train_and_eval(orig_df, "microsoft/deberta-v3-base", 0.7, 0.3, 1.5, "orig_deberta_dist")
    r.update({"name": "DeBERTa distilled", "dataset": "original", "distilled": "Yes"})
    all_results.append(r)
    print(f"    -> acc={r['test_acc']:.4f}  f1={r['test_f1']:.4f}  auroc={r['test_auroc']:.4f}")

    # ---- HARDENED DATASET ----
    print("\n" + "=" * 70)
    print("HARDENED DATASET (contrastive pairs + partial style normalization)")
    print("=" * 70)
    hard_df = pd.read_parquet(HARDENED_PATH)
    # Use hard labels as teacher soft targets for contrastive samples
    if "teacher_prob_tom" not in hard_df.columns:
        hard_df["teacher_prob_tom"] = hard_df["requires_tom"].astype(float)
    hard_df["teacher_prob_tom"] = hard_df["teacher_prob_tom"].fillna(
        hard_df["requires_tom"].astype(float)
    )
    print(f"  {len(hard_df)} samples\n")

    hard_baselines = run_baselines(hard_df, "hardened")
    all_results.extend(hard_baselines)
    for r in hard_baselines:
        print(f"  {r['name']:<30} acc={r['test_acc']:.4f}  f1={r['test_f1']:.4f}  auroc={r['test_auroc']:.4f}")

    # BERT-tiny hard vs distilled on hardened
    print("\n  BERT-tiny (4M) — hard only...")
    r = train_and_eval(hard_df, "google/bert_uncased_L-2_H-128_A-2", 1.0, 0.0, 1.5, "hard_tiny_hard")
    r.update({"name": "BERT-tiny hard", "dataset": "hardened", "distilled": "No"})
    all_results.append(r)
    print(f"    -> acc={r['test_acc']:.4f}  f1={r['test_f1']:.4f}  auroc={r['test_auroc']:.4f}")

    print("  BERT-tiny (4M) — distilled...")
    r = train_and_eval(hard_df, "google/bert_uncased_L-2_H-128_A-2", 0.7, 0.3, 1.5, "hard_tiny_dist")
    r.update({"name": "BERT-tiny distilled", "dataset": "hardened", "distilled": "Yes"})
    all_results.append(r)
    print(f"    -> acc={r['test_acc']:.4f}  f1={r['test_f1']:.4f}  auroc={r['test_auroc']:.4f}")

    # DeBERTa hard vs distilled on hardened
    print("\n  DeBERTa-v3-base — hard only...")
    r = train_and_eval(hard_df, "microsoft/deberta-v3-base", 1.0, 0.0, 1.5, "hard_deberta_hard")
    r.update({"name": "DeBERTa hard", "dataset": "hardened", "distilled": "No"})
    all_results.append(r)
    print(f"    -> acc={r['test_acc']:.4f}  f1={r['test_f1']:.4f}  auroc={r['test_auroc']:.4f}")

    print("  DeBERTa-v3-base — distilled...")
    r = train_and_eval(hard_df, "microsoft/deberta-v3-base", 0.7, 0.3, 1.5, "hard_deberta_dist")
    r.update({"name": "DeBERTa distilled", "dataset": "hardened", "distilled": "Yes"})
    all_results.append(r)
    print(f"    -> acc={r['test_acc']:.4f}  f1={r['test_f1']:.4f}  auroc={r['test_auroc']:.4f}")

    # ---- SUMMARY ----
    print("\n" + "=" * 70)
    print("COMPARATIVE SUMMARY: ORIGINAL vs HARDENED")
    print("=" * 70)
    print(f"\n{'Dataset':<12} {'Model':<25} {'Dist':>5} {'Acc':>8} {'F1':>8} {'AUROC':>8}")
    print("-" * 75)
    for r in all_results:
        auroc = f"{r['test_auroc']:.4f}" if isinstance(r.get('test_auroc'), (int, float)) else "-"
        dist = r.get('distilled', '-')
        print(f"{r.get('dataset',''):<12} {r['name']:<25} {dist:>5} "
              f"{r['test_acc']:>8.4f} {r['test_f1']:>8.4f} {auroc:>8}")

    # Save
    with open(OUTPUT_DIR / "hardened_ablation.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Markdown report
    lines = ["# Hardened Dataset Ablation\n"]
    lines.append("## Goal")
    lines.append("Compare original dataset (with source shortcuts) vs hardened dataset")
    lines.append("(contrastive pairs + style normalization) to show:\n")
    lines.append("1. Contrastive augmentation breaks source-only shortcuts")
    lines.append("2. Knowledge distillation provides larger gains on harder data")
    lines.append("3. Weak models benefit more from distillation on harder tasks\n")
    lines.append("## Results\n")
    lines.append("| Dataset | Model | Distilled | Acc | F1 | AUROC |")
    lines.append("|---------|-------|-----------|-----|----|----- |")
    for r in all_results:
        auroc = f"{r['test_auroc']:.4f}" if isinstance(r.get('test_auroc'), (int, float)) else "-"
        lines.append(
            f"| {r.get('dataset','')} | {r['name']} | {r.get('distilled','-')} | "
            f"{r['test_acc']:.4f} | {r['test_f1']:.4f} | {auroc} |"
        )

    with open(OUTPUT_DIR / "hardened_ablation.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
