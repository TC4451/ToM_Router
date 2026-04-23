"""Build multi-turn dialogue scenarios from the test set.

Creates four types of conversations:
1. Pure ToM dialogues (all turns require belief reasoning)
2. Pure social dialogues (all turns are non-ToM)
3. Mixed dialogues (alternating ToM and non-ToM)
4. Transition dialogues (start social, shift to ToM mid-conversation)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.seed import set_seed

DATASET_PATH = Path("data/processed/router_dataset_hardened.parquet")
OUTPUT_PATH = Path("data/processed/dialogue_scenarios.json")

SEED = 42


def build_scenarios(df):
    """Build structured multi-turn dialogue scenarios."""
    rng = np.random.RandomState(SEED)
    test = df[df["split"] == "test"].copy()
    tom = test[test["requires_tom"] == 1].reset_index(drop=True)
    non_tom = test[test["requires_tom"] == 0].reset_index(drop=True)

    scenarios = []
    tom_idx = 0
    non_tom_idx = 0

    # Shuffle deterministically
    tom = tom.sample(frac=1, random_state=SEED).reset_index(drop=True)
    non_tom = non_tom.sample(frac=1, random_state=SEED).reset_index(drop=True)

    def next_tom():
        nonlocal tom_idx
        if tom_idx >= len(tom):
            tom_idx = 0
        row = tom.iloc[tom_idx]
        tom_idx += 1
        return row

    def next_non_tom():
        nonlocal non_tom_idx
        if non_tom_idx >= len(non_tom):
            non_tom_idx = 0
        row = non_tom.iloc[non_tom_idx]
        non_tom_idx += 1
        return row

    def row_to_turn(row):
        return {
            "user_message": f"{row['context']} {row['question']}",
            "question": row["question"],
            "context": row["context"],
            "gold_answer": row.get("answer", ""),
            "requires_tom": int(row["requires_tom"]),
            "expected_route": "tom" if row["requires_tom"] == 1 else "social",
            "source": row["source_dataset"],
            "sample_id": row["sample_id"],
        }

    # Type 1: Pure ToM (10 scenarios, 6 turns each)
    for i in range(10):
        turns = [row_to_turn(next_tom()) for _ in range(6)]
        scenarios.append({
            "scenario_id": f"pure_tom_{i}",
            "type": "pure_tom",
            "description": "All turns require Theory of Mind reasoning",
            "turns": turns,
            "transition_turn": None,
        })

    # Type 2: Pure Social (10 scenarios, 6 turns each)
    for i in range(10):
        turns = [row_to_turn(next_non_tom()) for _ in range(6)]
        scenarios.append({
            "scenario_id": f"pure_social_{i}",
            "type": "pure_social",
            "description": "All turns are social reasoning (no ToM needed)",
            "turns": turns,
            "transition_turn": None,
        })

    # Type 3: Mixed (10 scenarios, 8 turns, alternating)
    for i in range(10):
        turns = []
        for j in range(8):
            if j % 2 == 0:
                turns.append(row_to_turn(next_tom()))
            else:
                turns.append(row_to_turn(next_non_tom()))
        scenarios.append({
            "scenario_id": f"mixed_{i}",
            "type": "mixed",
            "description": "Alternating ToM and non-ToM turns",
            "turns": turns,
            "transition_turn": None,
        })

    # Type 4: Transition social->ToM (10 scenarios, 8 turns)
    for i in range(10):
        transition = rng.randint(3, 6)  # transition point
        turns = []
        for j in range(8):
            if j < transition:
                turns.append(row_to_turn(next_non_tom()))
            else:
                turns.append(row_to_turn(next_tom()))
        scenarios.append({
            "scenario_id": f"transition_s2t_{i}",
            "type": "transition_social_to_tom",
            "description": f"Starts social, transitions to ToM at turn {transition}",
            "turns": turns,
            "transition_turn": transition,
        })

    # Type 5: Transition ToM->social (10 scenarios, 8 turns)
    for i in range(10):
        transition = rng.randint(3, 6)
        turns = []
        for j in range(8):
            if j < transition:
                turns.append(row_to_turn(next_tom()))
            else:
                turns.append(row_to_turn(next_non_tom()))
        scenarios.append({
            "scenario_id": f"transition_t2s_{i}",
            "type": "transition_tom_to_social",
            "description": f"Starts ToM, transitions to social at turn {transition}",
            "turns": turns,
            "transition_turn": transition,
        })

    return scenarios


def main():
    set_seed(SEED)

    print("Loading dataset...")
    df = pd.read_parquet(DATASET_PATH)
    print(f"  Test set: {len(df[df['split']=='test'])} samples")

    scenarios = build_scenarios(df)

    # Stats
    type_counts = {}
    total_turns = 0
    for s in scenarios:
        t = s["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
        total_turns += len(s["turns"])

    print(f"\nBuilt {len(scenarios)} scenarios ({total_turns} total turns):")
    for t, n in type_counts.items():
        print(f"  {t}: {n} scenarios")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(scenarios, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
