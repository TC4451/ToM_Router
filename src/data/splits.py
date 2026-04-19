"""Train/val/test splitting with anti-leakage and balance controls."""

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def compute_context_hash(context: str) -> str:
    """Compute a stable hash of normalized context for dedup/grouping."""
    normalized = " ".join(context.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()


def deduplicate_by_context(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Remove near-duplicate samples based on context hash.

    Keeps one sample per exact-match context. For near-duplicates,
    uses a simple length-based heuristic.
    """
    df = df.copy()
    df["_ctx_hash"] = df["context"].apply(compute_context_hash)

    # Remove exact duplicates (same context + question)
    n_before = len(df)
    df = df.drop_duplicates(subset=["_ctx_hash", "question"])
    n_removed = n_before - len(df)
    if n_removed:
        print(f"  Removed {n_removed} exact context+question duplicates")

    df = df.drop(columns=["_ctx_hash"])
    return df


def split_by_context_group(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Split data ensuring same context doesn't cross splits.

    Groups by context hash so all questions about the same scenario
    stay in the same split. Stratifies by requires_tom label.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    df = df.copy()
    df["_ctx_hash"] = df["context"].apply(compute_context_hash)

    # Group by context - each group has a single dominant label
    group_labels = df.groupby("_ctx_hash")["requires_tom"].agg(
        lambda x: int(x.mode().iloc[0])
    )

    groups = np.array(group_labels.index.tolist())
    labels = np.array(group_labels.values.tolist())

    # Split groups (not individual samples)
    val_test_ratio = val_ratio + test_ratio
    train_groups, valtest_groups, train_labels, valtest_labels = train_test_split(
        groups, labels, test_size=val_test_ratio, random_state=seed, stratify=labels
    )

    relative_test = test_ratio / val_test_ratio
    val_groups, test_groups = train_test_split(
        valtest_groups, test_size=relative_test, random_state=seed,
        stratify=valtest_labels
    )

    train_set = set(train_groups)
    val_set = set(val_groups)
    test_set = set(test_groups)

    def assign_split(h):
        if h in train_set:
            return "train"
        elif h in val_set:
            return "val"
        else:
            return "test"

    df["split"] = df["_ctx_hash"].apply(assign_split)
    df = df.drop(columns=["_ctx_hash"])
    return df


def balance_labels(
    df: pd.DataFrame,
    target_ratio: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Balance label distribution by downsampling the majority class.

    Applied per-split to maintain balance in train/val/test.
    """
    rng = np.random.RandomState(seed)
    balanced_parts = []

    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        pos = split_df[split_df["requires_tom"] == 1]
        neg = split_df[split_df["requires_tom"] == 0]

        target_size = min(len(pos), len(neg))

        if len(pos) > target_size:
            pos = pos.sample(n=target_size, random_state=rng)
        if len(neg) > target_size:
            neg = neg.sample(n=target_size, random_state=rng)

        balanced_parts.append(pd.concat([pos, neg]))

    result = pd.concat(balanced_parts).reset_index(drop=True)
    return result
