"""Generate teacher labels using OLMo-3-7B-Instruct.

Loads the processed router dataset, runs OLMo-3 teacher on each sample
using batched generation for speed, caches results incrementally, and
saves the labeled dataset.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.router_teacher import OLMoTeacherRouter
from src.utils.seed import set_seed

DATASET_PATH = Path("data/processed/router_dataset.parquet")
CACHE_PATH = Path("outputs/teacher_labels/teacher_cache.jsonl")
OUTPUT_PATH = Path("outputs/teacher_labels/router_dataset_with_teacher.parquet")

BATCH_SIZE = 4
LOG_EVERY = 50


def load_cache(path: Path) -> dict:
    """Load cached teacher labels."""
    cache = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    cache[rec["sample_id"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue
    return cache


def append_cache_batch(path: Path, records: list[dict]):
    """Append a batch of records to cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/teacher.yaml")
    parser.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    set_seed(42)

    # Load dataset
    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_parquet(DATASET_PATH)
    print(f"  {len(df)} samples")

    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"  Limited to {len(df)} samples")

    # Load cache
    cache = load_cache(CACHE_PATH)
    print(f"  {len(cache)} samples already cached")

    # Filter uncached
    uncached_mask = ~df["sample_id"].isin(cache.keys())
    uncached_df = df[uncached_mask].reset_index(drop=True)
    print(f"  {len(uncached_df)} samples to process")

    if len(uncached_df) == 0:
        print("All samples already cached, merging...")
    else:
        # Load teacher model
        print(f"\nLoading teacher model: {args.model}...")
        teacher = OLMoTeacherRouter(
            model_name=args.model,
            load_in_4bit=not args.no_4bit,
        )
        print("  Model loaded")

        # Process in batches
        start_time = time.time()
        total_processed = 0
        bs = args.batch_size

        for batch_start in range(0, len(uncached_df), bs):
            batch_end = min(batch_start + bs, len(uncached_df))
            batch_df = uncached_df.iloc[batch_start:batch_end]

            contexts = batch_df["context"].tolist()
            questions = batch_df["question"].tolist()
            sample_ids = batch_df["sample_id"].tolist()

            try:
                results = teacher.predict_batch(contexts, questions, batch_size=bs)

                # Attach sample IDs and cache
                records = []
                for sid, result in zip(sample_ids, results):
                    result["sample_id"] = sid
                    cache[sid] = result
                    records.append(result)

                append_cache_batch(CACHE_PATH, records)
                total_processed += len(records)

            except Exception as e:
                print(f"  BATCH ERROR at {batch_start}: {e}")
                # Fall back to single-sample processing
                for _, row in batch_df.iterrows():
                    try:
                        result = teacher.predict(row["context"], row["question"])
                    except Exception as e2:
                        result = {
                            "teacher_label": 0,
                            "teacher_prob_tom": 0.5,
                            "teacher_rationale": f"error: {str(e2)[:80]}",
                        }
                    result["sample_id"] = row["sample_id"]
                    cache[row["sample_id"]] = result
                    append_cache_batch(CACHE_PATH, [result])
                    total_processed += 1

            if total_processed % LOG_EVERY < bs:
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                remaining = (len(uncached_df) - total_processed) / rate if rate > 0 else 0
                print(
                    f"  [{total_processed}/{len(uncached_df)}] "
                    f"{rate:.2f} samples/sec, "
                    f"~{remaining/60:.0f} min remaining"
                )

        elapsed = time.time() - start_time
        print(f"\nProcessed {total_processed} samples in {elapsed/60:.1f} min "
              f"({total_processed/elapsed:.2f} samples/sec)")

        # Free GPU memory
        del teacher
        torch.cuda.empty_cache()

    # Merge teacher labels into dataset
    print("\nMerging teacher labels into dataset...")
    teacher_df = pd.DataFrame(list(cache.values()))

    df = df.merge(
        teacher_df[["sample_id", "teacher_label", "teacher_prob_tom", "teacher_rationale"]],
        on="sample_id",
        how="left",
        suffixes=("_orig", ""),
    )

    # Use teacher values, fall back to originals
    for col in ["teacher_label", "teacher_prob_tom", "teacher_rationale"]:
        orig_col = f"{col}_orig"
        if orig_col in df.columns:
            df[col] = df[col].fillna(df[orig_col])
            df = df.drop(columns=[orig_col])

    # Fill any remaining NaN
    df["teacher_label"] = df["teacher_label"].fillna(0).astype(int)
    df["teacher_prob_tom"] = df["teacher_prob_tom"].fillna(0.5).astype(float)
    df["teacher_rationale"] = df["teacher_rationale"].fillna("")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

    # Report agreement
    agree = (df["requires_tom"] == df["teacher_label"]).mean()
    print(f"\nTeacher-hard label agreement: {agree:.1%}")
    print(f"Teacher label distribution: {df['teacher_label'].value_counts().to_dict()}")
    print(f"Teacher prob_tom mean: {df['teacher_prob_tom'].mean():.3f}")

    # Disagreement analysis
    print("\n--- Disagreement Analysis ---")
    for h, t in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        mask = (df["requires_tom"] == h) & (df["teacher_label"] == t)
        print(f"  hard={h}, teacher={t}: {mask.sum()} samples")


if __name__ == "__main__":
    main()
