"""Prepare tasksource/tomi-nli dataset into unified schema.

Classic Sally-Anne style false belief tracking in NLI format.
All samples are ToM-positive (requires_tom=1).
Context = premise (story about agents moving objects).
Question = "Does this follow? " + hypothesis.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.schemas import RouterSample
from src.data.cleaners import clean_text

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/interim/tomi_nli.parquet")


def main():
    # Load all splits
    dfs = []
    for split in ["train", "validation", "test"]:
        path = RAW_DIR / f"tomi_nli_{split}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["_orig_split"] = split
            dfs.append(df)
            print(f"  {split}: {len(df)} samples")
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total raw: {len(df)} samples")

    rows = []
    for idx, row in df.iterrows():
        premise = clean_text(str(row["premise"]))
        hypothesis = clean_text(str(row["hypothesis"]))

        # Frame as context + question
        context = premise
        question = f"Does this follow? {hypothesis}"
        answer = str(row["label"])  # "entailment" or "not_entailment"

        sample = RouterSample(
            sample_id=f"tomi_nli_{idx}",
            source_dataset="tomi_nli",
            context=context,
            question=question,
            answer=answer,
            requires_tom=1,
            subtype="belief",
            original_category="false_belief_nli",
            metadata={
                "raw_id": str(idx),
                "nli_label": row["label"],
                "orig_split": row["_orig_split"],
            },
        )
        rows.append(sample.to_dict())

    result = pd.DataFrame(rows)

    # Drop empty
    result = result[result["context"].str.len() > 0]
    result = result[result["question"].str.len() > 0]

    # Deduplicate
    before = len(result)
    result = result.drop_duplicates(subset=["context", "question"])
    after = len(result)
    if before != after:
        print(f"  Removed {before - after} duplicates")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT_PATH, index=False)

    print(f"\nSaved {len(result)} samples to {OUT_PATH}")
    print(f"Avg context length: {result['context'].str.len().mean():.0f} chars")
    print(f"Avg question length: {result['question'].str.len().mean():.0f} chars")


if __name__ == "__main__":
    main()
