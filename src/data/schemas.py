"""Unified data schema for the ToM router dataset."""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class RouterSample:
    """One sample in the unified router dataset."""
    sample_id: str
    source_dataset: str  # "simpletom" or "kokomind"
    context: str
    question: str
    answer: Optional[str] = None
    requires_tom: int = 0  # 0 or 1
    subtype: str = "other"  # belief, emotion, norm, relation, counterfactual, advice, other
    original_category: Optional[str] = None
    split: str = "train"
    teacher_prob_tom: float = 0.0
    teacher_label: int = 0
    teacher_rationale: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


REQUIRED_FIELDS = [
    "sample_id", "source_dataset", "context", "question",
    "requires_tom", "subtype",
]

KOKOMIND_CATEGORY_MAP = {
    "ToM": {"requires_tom": 1, "subtype": "belief"},
    "Emotion Recognition": {"requires_tom": 0, "subtype": "emotion"},
    "Social Norm": {"requires_tom": 0, "subtype": "norm"},
    "Social Relation": {"requires_tom": 0, "subtype": "relation"},
    "Counterfactual": {"requires_tom": 0, "subtype": "counterfactual"},
    "Social Advice": {"requires_tom": 0, "subtype": "advice"},
}

SIMPLETOM_QA_TYPES = {
    "behavior-qa": "behavior",
    "judgment-qa": "judgment",
    "mental-state-qa": "mental_state",
}

# All source datasets in the project
ALL_SOURCES = [
    "simpletom",
    "kokomind",
    "theory_of_mind",
    "tomi_nli",
    "social_iqa",
    "cicero",
]
