"""Build hardened dataset by combining:
1. Original samples (style-normalized where available, original otherwise)
2. Contrastive question pairs (same context, opposite label)

Then re-split and re-balance.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.splits import deduplicate_by_context, split_by_context_group, balance_labels
from src.data.cleaners import clean_text

ORIGINAL_PATH = Path("data/processed/router_dataset.parquet")
CONTRASTIVE_PATH = Path("outputs/contrastive/contrastive_questions.parquet")
STYLE_CACHE_PATH = Path("outputs/style_norm/style_cache.jsonl")
OUT_PATH = Path("data/processed/router_dataset_hardened.parquet")
REPORT_PATH = Path("outputs/reports/hardened_dataset_report.txt")

SEED = 42


def load_style_cache():
    """Load whatever style normalization we have."""
    import json
    cache = {}
    if STYLE_CACHE_PATH.exists():
        with open(STYLE_CACHE_PATH) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "ok":
                        cache[rec["sample_id"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue
    return cache


def main():
    np.random.seed(SEED)

    print("Loading datasets...")
    orig_df = pd.read_parquet(ORIGINAL_PATH)
    contrastive_df = pd.read_parquet(CONTRASTIVE_PATH)
    style_cache = load_style_cache()
    print(f"  Original: {len(orig_df)} samples")
    print(f"  Contrastive: {len(contrastive_df)} samples")
    print(f"  Style-normalized: {len(style_cache)} samples")

    # Apply style normalization where available
    norm_applied = 0
    for idx, row in orig_df.iterrows():
        if row["sample_id"] in style_cache:
            sc = style_cache[row["sample_id"]]
            orig_df.at[idx, "context"] = clean_text(sc["norm_context"])
            orig_df.at[idx, "question"] = clean_text(sc["norm_question"])
            norm_applied += 1
    print(f"  Applied style normalization to {norm_applied} original samples")

    # Clean contrastive samples
    contrastive_df = contrastive_df[contrastive_df["requires_tom"].isin([0, 1])].copy()
    contrastive_df["context"] = contrastive_df["context"].apply(clean_text)
    contrastive_df["question"] = contrastive_df["question"].apply(clean_text)
    print(f"  Valid contrastive: {len(contrastive_df)}")

    # Ensure contrastive samples have teacher fields
    for col in ["teacher_prob_tom", "teacher_label", "teacher_rationale"]:
        if col not in contrastive_df.columns:
            if col == "teacher_prob_tom":
                contrastive_df[col] = contrastive_df["requires_tom"].astype(float)
            elif col == "teacher_label":
                contrastive_df[col] = contrastive_df["requires_tom"]
            else:
                contrastive_df[col] = "contrastive_generated"

    # Merge original + contrastive
    merged = pd.concat([orig_df, contrastive_df], ignore_index=True)
    print(f"\nMerged: {len(merged)} samples")
    print(f"  ToM: {(merged['requires_tom'] == 1).sum()}")
    print(f"  Non-ToM: {(merged['requires_tom'] == 0).sum()}")

    # Deduplicate
    print("\nDeduplicating...")
    n_before = len(merged)
    merged = merged.drop_duplicates(subset=["context", "question"])
    print(f"  Removed {n_before - len(merged)} duplicates")
    print(f"  After dedup: {len(merged)} samples")

    # Re-split by context group
    print("\nRe-splitting train/val/test...")
    merged = split_by_context_group(merged, seed=SEED)

    # Balance
    print("\nBalancing labels per split...")
    merged = balance_labels(merged, seed=SEED)
    print(f"After balancing: {len(merged)} samples")

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")

    # Report
    lines = []
    lines.append("=" * 60)
    lines.append("HARDENED DATASET REPORT")
    lines.append("=" * 60)
    lines.append(f"\nTotal samples: {len(merged)}")

    lines.append(f"\n--- Label by source ---")
    ct = pd.crosstab(merged["source_dataset"], merged["requires_tom"], margins=True)
    lines.append(ct.to_string())

    lines.append(f"\n--- Split distribution ---")
    split_ct = pd.crosstab(merged["split"], merged["requires_tom"], margins=True)
    lines.append(split_ct.to_string())

    lines.append(f"\n--- Contrastive coverage ---")
    is_contrastive = merged["source_dataset"].str.contains("contrastive", na=False)
    lines.append(f"  Contrastive samples: {is_contrastive.sum()}")
    lines.append(f"  Original samples: {(~is_contrastive).sum()}")

    lines.append(f"\n--- Source diversity per label ---")
    for label in [0, 1]:
        subset = merged[merged["requires_tom"] == label]
        lines.append(f"  requires_tom={label}:")
        lines.append(f"    {subset['source_dataset'].value_counts().to_dict()}")

    lines.append(f"\n--- Context length by label ---")
    for label in [0, 1]:
        lens = merged[merged["requires_tom"] == label]["context"].str.len()
        lines.append(f"  requires_tom={label}: mean={lens.mean():.0f}, median={lens.median():.0f}")

    report = "\n".join(lines)
    print(report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
