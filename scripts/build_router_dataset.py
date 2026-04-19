"""Build the merged, balanced, and split router dataset.

Merges all interim datasets (SimpleToM, KokoMind, theory_of_mind, tomi_nli,
social_iqa, cicero), applies deduplication, subsamples to target size,
balances labels, splits into train/val/test, and generates a report.

Target: 5k-10k total samples, roughly balanced ToM vs Non-ToM.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.splits import deduplicate_by_context, split_by_context_group, balance_labels

INTERIM_DIR = Path("data/interim")
OUT_PATH = Path("data/processed/router_dataset.parquet")
REPORT_PATH = Path("outputs/reports/dataset_report.txt")

SEED = 42
TARGET_PER_CLASS = 4000  # ~8000 total, within 5k-10k range

# All interim datasets to load
DATASETS = [
    "simpletom",
    "kokomind",
    "theory_of_mind",
    "tomi_nli",
    "social_iqa",
    "cicero",
]


def generate_report(df: pd.DataFrame) -> str:
    """Generate dataset statistics report."""
    lines = []
    lines.append("=" * 60)
    lines.append("ROUTER DATASET REPORT")
    lines.append("=" * 60)

    lines.append(f"\nTotal samples: {len(df)}")

    # Label counts by source
    lines.append("\n--- Label counts by source ---")
    ct = pd.crosstab(df["source_dataset"], df["requires_tom"], margins=True)
    lines.append(ct.to_string())

    # Subtype distribution
    lines.append("\n--- Subtype distribution ---")
    lines.append(df["subtype"].value_counts().to_string())

    # Split distribution
    lines.append("\n--- Split distribution ---")
    split_ct = pd.crosstab(df["split"], df["requires_tom"], margins=True)
    lines.append(split_ct.to_string())

    # Source mix by split
    lines.append("\n--- Source mix by split ---")
    src_split = pd.crosstab(df["split"], df["source_dataset"], margins=True)
    lines.append(src_split.to_string())

    # Average context length by label
    lines.append("\n--- Average context length (chars) by label ---")
    ctx_len = df.groupby("requires_tom")["context"].apply(lambda x: x.str.len().mean())
    lines.append(ctx_len.to_string())

    # Average question length by label
    lines.append("\n--- Average question length (chars) by label ---")
    q_len = df.groupby("requires_tom")["question"].apply(lambda x: x.str.len().mean())
    lines.append(q_len.to_string())

    # Length distribution check (potential shortcut)
    lines.append("\n--- Length distribution check ---")
    df_temp = df.copy()
    df_temp["ctx_len"] = df_temp["context"].str.len()
    for label in [0, 1]:
        subset = df_temp[df_temp["requires_tom"] == label]["ctx_len"]
        lines.append(
            f"  requires_tom={label}: mean={subset.mean():.0f}, "
            f"median={subset.median():.0f}, std={subset.std():.0f}, "
            f"min={subset.min()}, max={subset.max()}"
        )

    # Source diversity per label
    lines.append("\n--- Source diversity per label ---")
    for label in [0, 1]:
        subset = df[df["requires_tom"] == label]
        lines.append(f"  requires_tom={label}:")
        lines.append(f"    {subset['source_dataset'].value_counts().to_dict()}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def subsample_to_target(
    df: pd.DataFrame, target_per_class: int, seed: int
) -> pd.DataFrame:
    """Subsample each class to target size, preserving source diversity.

    For each label, sample proportionally from each source dataset.
    """
    rng = np.random.RandomState(seed)
    parts = []

    for label in [0, 1]:
        class_df = df[df["requires_tom"] == label]

        if len(class_df) <= target_per_class:
            parts.append(class_df)
            print(f"  requires_tom={label}: keeping all {len(class_df)} (under target {target_per_class})")
            continue

        # Proportional sampling from each source
        source_counts = class_df["source_dataset"].value_counts()
        total = len(class_df)
        sampled = []

        for source, count in source_counts.items():
            n_sample = max(1, int(target_per_class * count / total))
            source_df = class_df[class_df["source_dataset"] == source]
            n_sample = min(n_sample, len(source_df))
            sampled.append(source_df.sample(n=n_sample, random_state=rng))

        combined = pd.concat(sampled)

        # Adjust if we're over/under target
        if len(combined) > target_per_class:
            combined = combined.sample(n=target_per_class, random_state=rng)
        elif len(combined) < target_per_class:
            remaining = class_df[~class_df.index.isin(combined.index)]
            extra = min(target_per_class - len(combined), len(remaining))
            if extra > 0:
                combined = pd.concat([
                    combined,
                    remaining.sample(n=extra, random_state=rng)
                ])

        parts.append(combined)
        print(f"  requires_tom={label}: sampled {len(combined)} from {total}")

    return pd.concat(parts).reset_index(drop=True)


def main():
    np.random.seed(SEED)

    # Load all interim datasets
    print("Loading interim datasets...")
    all_dfs = []
    for name in DATASETS:
        path = INTERIM_DIR / f"{name}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            print(f"  {name}: {len(df)} samples "
                  f"(ToM={int((df['requires_tom'] == 1).sum())}, "
                  f"Non-ToM={int((df['requires_tom'] == 0).sum())})")
            all_dfs.append(df)
        else:
            print(f"  {name}: NOT FOUND at {path}, skipping")

    # Merge
    merged = pd.concat(all_dfs, ignore_index=True)
    print(f"\nMerged: {len(merged)} samples")
    print(f"  ToM: {(merged['requires_tom'] == 1).sum()}")
    print(f"  Non-ToM: {(merged['requires_tom'] == 0).sum()}")

    # Deduplicate
    print("\nDeduplicating...")
    merged = deduplicate_by_context(merged)
    print(f"After dedup: {len(merged)} samples")

    # Subsample to target size
    print(f"\nSubsampling to ~{TARGET_PER_CLASS * 2} total...")
    merged = subsample_to_target(merged, TARGET_PER_CLASS, SEED)
    print(f"After subsampling: {len(merged)} samples")

    # Split by context group (anti-leakage)
    print("\nSplitting train/val/test...")
    merged = split_by_context_group(merged, seed=SEED)

    # Balance labels within each split
    print("\nBalancing labels per split...")
    merged = balance_labels(merged, seed=SEED)
    print(f"After balancing: {len(merged)} samples")

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")

    # Generate report
    report = generate_report(merged)
    print(report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
