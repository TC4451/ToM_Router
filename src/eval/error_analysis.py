"""Error analysis for router predictions."""

import numpy as np
import pandas as pd


def generate_error_report(
    df: pd.DataFrame,
    probs: np.ndarray,
    threshold: float = 0.5,
    n_examples: int = 10,
) -> str:
    """Generate a detailed error analysis report."""
    df = df.copy()
    df["student_prob"] = probs
    df["student_pred"] = (probs >= threshold).astype(int)
    df["correct"] = df["requires_tom"] == df["student_pred"]

    lines = []
    lines.append("=" * 70)
    lines.append("ERROR ANALYSIS REPORT")
    lines.append("=" * 70)

    # False negatives (missed ToM)
    fn = df[(df["requires_tom"] == 1) & (df["student_pred"] == 0)]
    lines.append(f"\n## FALSE NEGATIVES (missed ToM): {len(fn)} samples")
    lines.append("These are ToM questions incorrectly routed to the social expert.")
    for _, row in fn.head(n_examples).iterrows():
        lines.append(f"\n  [{row['sample_id']}] source={row['source_dataset']}, subtype={row['subtype']}")
        lines.append(f"  prob_tom={row['student_prob']:.3f}")
        if "teacher_prob_tom" in row and pd.notna(row.get("teacher_prob_tom")):
            lines.append(f"  teacher_prob={row['teacher_prob_tom']:.3f}")
        lines.append(f"  context: {row['context'][:150]}...")
        lines.append(f"  question: {row['question'][:100]}")

    # False positives (unnecessary ToM)
    fp = df[(df["requires_tom"] == 0) & (df["student_pred"] == 1)]
    lines.append(f"\n## FALSE POSITIVES (unnecessary ToM): {len(fp)} samples")
    lines.append("These are non-ToM questions incorrectly routed to the ToM expert.")
    for _, row in fp.head(n_examples).iterrows():
        lines.append(f"\n  [{row['sample_id']}] source={row['source_dataset']}, subtype={row['subtype']}")
        lines.append(f"  prob_tom={row['student_prob']:.3f}")
        if "teacher_prob_tom" in row and pd.notna(row.get("teacher_prob_tom")):
            lines.append(f"  teacher_prob={row['teacher_prob_tom']:.3f}")
        lines.append(f"  context: {row['context'][:150]}...")
        lines.append(f"  question: {row['question'][:100]}")

    # Near decision boundary
    boundary = df[np.abs(df["student_prob"] - threshold) < 0.1]
    lines.append(f"\n## NEAR DECISION BOUNDARY: {len(boundary)} samples")
    for _, row in boundary.head(n_examples).iterrows():
        lines.append(f"\n  [{row['sample_id']}] label={row['requires_tom']}, pred={row['student_pred']}")
        lines.append(f"  prob_tom={row['student_prob']:.3f}, source={row['source_dataset']}")
        lines.append(f"  context: {row['context'][:120]}...")
        lines.append(f"  question: {row['question'][:80]}")

    # Teacher-student disagreement
    if "teacher_label" in df.columns:
        disagree = df[df["student_pred"] != df["teacher_label"]]
        lines.append(f"\n## TEACHER-STUDENT DISAGREEMENT: {len(disagree)} samples")
        for _, row in disagree.head(n_examples).iterrows():
            lines.append(
                f"\n  [{row['sample_id']}] hard={row['requires_tom']}, "
                f"teacher={row.get('teacher_label', '?')}, student={row['student_pred']}"
            )
            lines.append(f"  student_prob={row['student_prob']:.3f}, teacher_prob={row.get('teacher_prob_tom', '?')}")
            lines.append(f"  context: {row['context'][:120]}...")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
