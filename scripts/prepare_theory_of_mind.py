"""Prepare hmamin/theory_of_mind dataset into unified schema.

All samples are ToM-positive (requires_tom=1). Contains aggregated ToM benchmarks
from ToMBench and Hi-ToM covering false belief, faux pas, strange stories, etc.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.schemas import RouterSample
from src.data.cleaners import clean_text

RAW_PATH = Path("data/raw/theory_of_mind_train.parquet")
OUT_PATH = Path("data/interim/theory_of_mind.parquet")

# Map categories to subtypes
CATEGORY_SUBTYPE = {
    "false_belief_task": "belief",
    "faux_pas_recognition_test": "belief",
    "strange_story_task": "belief",
    "unexpected_outcome_test": "belief",
    "ambiguous_story_task": "belief",
    "scalar_implicature_test": "belief",
    "hinting_task_test": "intention",
    "persuasion_story_task": "intention",
    "hidden_emotions": "emotion_tom",
    "moral_emotions": "emotion_tom",
    "discrepant_emotions": "emotion_tom",
    "discrepant_intentions": "intention",
    "discrepant_desires": "desire",
    "multiple_desires": "desire",
    "prediction_of_actions": "belief",
    "percepts_knowledge_links": "belief",
    "knowledge_pretend_play_links": "belief",
    "knowledge_attention_links": "belief",
    "emotion_regulation": "emotion_tom",
    "completion_of_failed_actions": "intention",
}


def main():
    print(f"Loading {RAW_PATH}...")
    df = pd.read_parquet(RAW_PATH)
    print(f"  {len(df)} raw samples")

    rows = []
    for _, row in df.iterrows():
        cat = str(row["category"]) if pd.notna(row["category"]) else "unknown"
        subtype = CATEGORY_SUBTYPE.get(cat, "belief")

        sample = RouterSample(
            sample_id=f"tom_{row.name}_{cat[:10]}",
            source_dataset="theory_of_mind",
            context=clean_text(str(row["story"])) if pd.notna(row["story"]) else "",
            question=clean_text(str(row["question"])) if pd.notna(row["question"]) else "",
            answer=clean_text(str(row["answer"])) if pd.notna(row["answer"]) else None,
            requires_tom=1,
            subtype=subtype,
            original_category=cat,
            metadata={
                "raw_id": str(row.name),
                "source": row.get("source", ""),
                "answer_letter": row.get("answer_letter", ""),
            },
        )
        rows.append(sample.to_dict())

    result = pd.DataFrame(rows)

    # Drop samples with empty context or question
    before = len(result)
    result = result[result["context"].str.len() > 0]
    result = result[result["question"].str.len() > 0]
    after = len(result)
    if before != after:
        print(f"  Dropped {before - after} samples with empty context/question")

    # Deduplicate
    before = len(result)
    result = result.drop_duplicates(subset=["context", "question"])
    after = len(result)
    if before != after:
        print(f"  Removed {before - after} duplicates")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT_PATH, index=False)

    print(f"\nSaved {len(result)} samples to {OUT_PATH}")
    print(f"Category distribution: {result['original_category'].value_counts().to_dict()}")
    print(f"Subtype distribution: {result['subtype'].value_counts().to_dict()}")
    print(f"Avg context length: {result['context'].str.len().mean():.0f} chars")
    print(f"Avg question length: {result['question'].str.len().mean():.0f} chars")


if __name__ == "__main__":
    main()
