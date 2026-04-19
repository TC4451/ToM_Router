"""Prepare KokoMind dataset into unified schema.

Loads raw KokoMind JSONL, splits text into context/question,
maps categories to ToM labels, and saves to data/interim/.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.schemas import RouterSample, KOKOMIND_CATEGORY_MAP
from src.data.cleaners import clean_text, split_kokomind_text

RAW_PATH = Path("data/raw/kokomind/question_nonverbal_yes_v0.1.jsonl")
OUT_PATH = Path("data/interim/kokomind.parquet")


def load_kokomind(path: Path) -> list[dict]:
    """Load KokoMind JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def process_kokomind(records: list[dict]) -> list[dict]:
    """Convert KokoMind records to unified schema."""
    rows = []
    skipped = 0

    for rec in records:
        category = rec["category"]
        if category not in KOKOMIND_CATEGORY_MAP:
            print(f"  Warning: unknown category '{category}', skipping")
            skipped += 1
            continue

        mapping = KOKOMIND_CATEGORY_MAP[category]
        context, question = split_kokomind_text(rec["text"])

        if not question:
            print(f"  Warning: empty question for id={rec['question_id']}, skipping")
            skipped += 1
            continue

        sample = RouterSample(
            sample_id=f"kokomind_{rec['question_id']}",
            source_dataset="kokomind",
            context=context,
            question=question,
            answer=clean_text(rec["answer"]),
            requires_tom=mapping["requires_tom"],
            subtype=mapping["subtype"],
            original_category=category,
            metadata={
                "raw_id": str(rec["question_id"]),
                "source": rec["source"],
            },
        )
        rows.append(sample.to_dict())

    if skipped:
        print(f"  Skipped {skipped} records")
    return rows


def main():
    print(f"Loading KokoMind from {RAW_PATH}...")
    records = load_kokomind(RAW_PATH)
    print(f"  Loaded {len(records)} raw records")

    print("Processing...")
    rows = process_kokomind(records)
    df = pd.DataFrame(rows)

    # Check for duplicates
    n_before = len(df)
    df = df.drop_duplicates(subset=["context", "question"])
    n_after = len(df)
    if n_before != n_after:
        print(f"Removed {n_before - n_after} exact duplicates")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print(f"\nSaved {len(df)} samples to {OUT_PATH}")
    print(f"Label distribution:")
    print(f"  requires_tom=1: {(df['requires_tom'] == 1).sum()}")
    print(f"  requires_tom=0: {(df['requires_tom'] == 0).sum()}")
    print(f"Category distribution: {df['original_category'].value_counts().to_dict()}")
    print(f"Subtype distribution: {df['subtype'].value_counts().to_dict()}")
    print(f"Avg context length: {df['context'].str.len().mean():.0f} chars")
    print(f"Avg question length: {df['question'].str.len().mean():.0f} chars")


if __name__ == "__main__":
    main()
