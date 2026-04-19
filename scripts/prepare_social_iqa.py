"""Prepare allenai/social_i_qa dataset into unified schema.

Social commonsense QA — non-ToM social reasoning (requires_tom=0).
Questions about motivations, reactions, next actions in social situations.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.schemas import RouterSample
from src.data.cleaners import clean_text

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/interim/social_iqa.parquet")

# Map question patterns to subtypes
QUESTION_SUBTYPE_PATTERNS = {
    "feel": "emotion",
    "emotion": "emotion",
    "react": "emotion",
    "mood": "emotion",
    "norm": "norm",
    "should": "norm",
    "rude": "norm",
    "polite": "norm",
    "relationship": "relation",
    "friend": "relation",
}


def classify_subtype(question: str) -> str:
    q_lower = question.lower()
    for pattern, subtype in QUESTION_SUBTYPE_PATTERNS.items():
        if pattern in q_lower:
            return subtype
    return "social_commonsense"


def main():
    dfs = []
    for split in ["train", "validation"]:
        path = RAW_DIR / f"social_iqa_{split}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["_orig_split"] = split
            dfs.append(df)
            print(f"  {split}: {len(df)} samples")
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total raw: {len(df)} samples")

    rows = []
    for idx, row in df.iterrows():
        context = clean_text(str(row["context"]))
        question = clean_text(str(row["question"]))

        # Get correct answer
        label_map = {"1": "answerA", "2": "answerB", "3": "answerC"}
        answer_col = label_map.get(str(row["label"]), "answerA")
        answer = clean_text(str(row[answer_col])) if pd.notna(row.get(answer_col)) else None

        subtype = classify_subtype(question)

        sample = RouterSample(
            sample_id=f"siqa_{idx}",
            source_dataset="social_iqa",
            context=context,
            question=question,
            answer=answer,
            requires_tom=0,
            subtype=subtype,
            original_category="social_commonsense_qa",
            metadata={
                "raw_id": str(idx),
                "orig_split": row["_orig_split"],
            },
        )
        rows.append(sample.to_dict())

    result = pd.DataFrame(rows)

    result = result[result["context"].str.len() > 0]
    result = result[result["question"].str.len() > 0]

    before = len(result)
    result = result.drop_duplicates(subset=["context", "question"])
    after = len(result)
    if before != after:
        print(f"  Removed {before - after} duplicates")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT_PATH, index=False)

    print(f"\nSaved {len(result)} samples to {OUT_PATH}")
    print(f"Subtype distribution: {result['subtype'].value_counts().to_dict()}")
    print(f"Avg context length: {result['context'].str.len().mean():.0f} chars")
    print(f"Avg question length: {result['question'].str.len().mean():.0f} chars")


if __name__ == "__main__":
    main()
