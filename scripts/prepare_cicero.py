"""Prepare declare-lab/cicero dataset into unified schema.

Commonsense inference in dialogues — non-ToM social reasoning (requires_tom=0).
Questions about causes, motivations, subsequent events, emotional reactions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.schemas import RouterSample
from src.data.cleaners import clean_text

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/interim/cicero.parquet")

QUESTION_SUBTYPE_MAP = {
    "cause": "social_commonsense",
    "motivation": "social_commonsense",
    "subsequent": "social_commonsense",
    "emotion": "emotion",
    "prerequisite": "social_commonsense",
    "react": "emotion",
}


def classify_subtype(question: str) -> str:
    q_lower = question.lower()
    for pattern, subtype in QUESTION_SUBTYPE_MAP.items():
        if pattern in q_lower:
            return subtype
    return "social_commonsense"


def format_dialogue(dialogue) -> str:
    """Convert dialogue field (list of strings) to readable text."""
    if isinstance(dialogue, list):
        return " ".join(str(turn).strip() for turn in dialogue)
    return str(dialogue)


def main():
    dfs = []
    for split in ["train", "validation", "test"]:
        path = RAW_DIR / f"cicero_{split}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["_orig_split"] = split
            dfs.append(df)
            print(f"  {split}: {len(df)} samples")
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total raw: {len(df)} samples")

    rows = []
    for idx, row in df.iterrows():
        dialogue_text = format_dialogue(row["Dialogue"])
        context = clean_text(dialogue_text)
        question = clean_text(str(row["Question"]))

        # Get correct answer from Choices using Correct Answers index
        answer = None
        try:
            choices = row["Choices"]
            correct = row["Correct Answers"]
            if correct is not None and choices is not None:
                if isinstance(correct, (list, np.ndarray)) and len(correct) > 0:
                    if isinstance(choices, (list, np.ndarray)) and len(choices) > 0:
                        idx_val = int(correct[0])
                        if 0 <= idx_val < len(choices):
                            answer = clean_text(str(choices[idx_val]))
        except (TypeError, ValueError, IndexError):
            pass

        subtype = classify_subtype(question)

        sample = RouterSample(
            sample_id=f"cicero_{row['ID']}",
            source_dataset="cicero",
            context=context,
            question=question,
            answer=answer,
            requires_tom=0,
            subtype=subtype,
            original_category="commonsense_dialogue",
            metadata={
                "raw_id": str(row["ID"]),
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
