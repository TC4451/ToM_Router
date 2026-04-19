"""Prepare SimpleToM dataset into unified schema.

Loads raw SimpleToM QA subsets (behavior-qa, judgment-qa, mental-state-qa),
normalizes them into the unified router schema, and saves to data/interim/.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.schemas import RouterSample, SIMPLETOM_QA_TYPES
from src.data.cleaners import clean_text, format_choices

RAW_DIR = Path("data/raw/simpletom")
OUT_PATH = Path("data/interim/simpletom.parquet")


def process_qa_subset(subset_name: str) -> list[dict]:
    """Process one SimpleToM QA subset into unified rows."""
    qa_type = SIMPLETOM_QA_TYPES[subset_name]
    path = RAW_DIR / f"{subset_name}_test.parquet"

    if not path.exists():
        print(f"  Warning: {path} not found, skipping")
        return []

    df = pd.read_parquet(path)
    rows = []

    for _, row in df.iterrows():
        answer = format_choices(row["choices"], row["answerKey"])

        sample = RouterSample(
            sample_id=f"simpletom_{row['id']}",
            source_dataset="simpletom",
            context=clean_text(row["story"]),
            question=clean_text(row["question"]),
            answer=answer,
            requires_tom=1,
            subtype="belief",
            original_category=subset_name,
            metadata={
                "raw_id": row["id"],
                "question_type": qa_type,
                "scenario_name": row["scenario_name"],
                "answer_key": row["answerKey"],
            },
        )
        rows.append(sample.to_dict())

    return rows


def main():
    all_rows = []

    for subset_name in SIMPLETOM_QA_TYPES:
        print(f"Processing {subset_name}...")
        rows = process_qa_subset(subset_name)
        print(f"  -> {len(rows)} samples")
        all_rows.extend(rows)

    # Deduplicate by (context, question) - same story appears across QA types
    # Keep all since they have different questions
    df = pd.DataFrame(all_rows)

    # Check for exact duplicates
    n_before = len(df)
    df = df.drop_duplicates(subset=["context", "question"])
    n_after = len(df)
    if n_before != n_after:
        print(f"Removed {n_before - n_after} exact duplicates")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print(f"\nSaved {len(df)} samples to {OUT_PATH}")
    print(f"Question types: {df['original_category'].value_counts().to_dict()}")
    print(f"Avg context length: {df['context'].str.len().mean():.0f} chars")
    print(f"Avg question length: {df['question'].str.len().mean():.0f} chars")


if __name__ == "__main__":
    main()
