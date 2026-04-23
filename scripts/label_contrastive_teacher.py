"""Run OLMo-3 teacher on contrastive samples to replace placeholder soft labels."""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.router_teacher import OLMoTeacherRouter
from src.utils.seed import set_seed

DATASET_PATH = Path("data/processed/router_dataset_hardened.parquet")
CACHE_PATH = Path("outputs/contrastive/contrastive_teacher_cache.jsonl")
OUTPUT_PATH = Path("data/processed/router_dataset_hardened_v2.parquet")

LOG_EVERY = 50


def load_cache(path):
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


def append_cache(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main():
    set_seed(42)

    print("Loading dataset...")
    df = pd.read_parquet(DATASET_PATH)

    # Find contrastive samples with placeholder labels
    mask = df["teacher_rationale"] == "contrastive_generated"
    contrastive = df[mask].reset_index(drop=True)
    print(f"  {len(contrastive)} contrastive samples need real teacher labels")

    cache = load_cache(CACHE_PATH)
    uncached = contrastive[~contrastive["sample_id"].isin(cache.keys())].reset_index(drop=True)
    print(f"  {len(cache)} already cached, {len(uncached)} to process")

    if len(uncached) > 0:
        print("\nLoading OLMo-3-7B-Instruct...")
        teacher = OLMoTeacherRouter(load_in_4bit=True)
        print("  Model loaded")

        start = time.time()
        for i, (_, row) in enumerate(uncached.iterrows()):
            try:
                result = teacher.predict(row["context"], row["question"])
            except Exception as e:
                result = {"teacher_label": int(row["requires_tom"]),
                          "teacher_prob_tom": float(row["requires_tom"]),
                          "teacher_rationale": f"error: {str(e)[:80]}"}
            result["sample_id"] = row["sample_id"]
            cache[row["sample_id"]] = result
            append_cache(CACHE_PATH, [result])

            if (i + 1) % LOG_EVERY == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                remaining = (len(uncached) - i - 1) / rate
                print(f"  [{i+1}/{len(uncached)}] {rate:.2f}/sec, ~{remaining/60:.0f}min left")

        elapsed = time.time() - start
        print(f"\nProcessed {len(uncached)} in {elapsed/60:.1f}min")
        del teacher
        torch.cuda.empty_cache()

    # Merge back into dataset
    print("\nMerging real teacher labels...")
    teacher_df = pd.DataFrame(list(cache.values()))

    for _, row in teacher_df.iterrows():
        idx = df.index[df["sample_id"] == row["sample_id"]]
        if len(idx) > 0:
            df.loc[idx, "teacher_prob_tom"] = row["teacher_prob_tom"]
            df.loc[idx, "teacher_label"] = row["teacher_label"]
            df.loc[idx, "teacher_rationale"] = row["teacher_rationale"]

    # Verify
    still_placeholder = (df["teacher_rationale"] == "contrastive_generated").sum()
    print(f"  Remaining placeholders: {still_placeholder}")

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"  Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
