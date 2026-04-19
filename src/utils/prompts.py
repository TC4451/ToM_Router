"""Prompt templates for teacher labeling."""

TEACHER_PROMPT_V1 = """You are labeling social reasoning tasks.

Given a context and a question, decide whether answering the question requires Theory of Mind reasoning.

Definition:
Theory of Mind is required when the answer depends on reasoning about a person's hidden beliefs, false beliefs, intentions, knowledge state, perspective, or information that is not directly observable in the scene.

Non-Theory-of-Mind means the question can be answered from observable facts, explicit emotional content, social norms, or relationship knowledge without inferring hidden mental states.

Return JSON with fields:
- requires_tom: 0 or 1
- prob_tom: float between 0 and 1
- rationale: one short sentence

Context: {context}
Question: {question}

Return ONLY valid JSON, no other text."""


def format_teacher_prompt(context: str, question: str, template: str = "tom_router_v1") -> str:
    """Format a teacher prompt with context and question."""
    if template == "tom_router_v1":
        return TEACHER_PROMPT_V1.format(context=context, question=question)
    raise ValueError(f"Unknown template: {template}")
