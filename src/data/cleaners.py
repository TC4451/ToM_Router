"""Text normalization utilities for dataset preprocessing."""

import re
import unicodedata


def clean_text(text: str) -> str:
    """Clean and normalize text while preserving semantics.

    - Strips excessive whitespace
    - Standardizes quotes and punctuation
    - Removes duplicate spaces and line breaks
    - Trims leading instruction artifacts
    """
    if not text:
        return ""

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Standardize quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2026", "...")

    # Collapse multiple whitespace/newlines into single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def split_kokomind_text(text: str) -> tuple[str, str]:
    """Split KokoMind text field into (context, question).

    Format: "Read the following context: <context>.. \n\n**Briefly answer these question**: \n<question>"
    """
    separator = "**Briefly answer these question**:"

    if separator in text:
        parts = text.split(separator, 1)
        context = parts[0].strip()
        question = parts[1].strip()
    else:
        # Fallback: treat entire text as context
        context = text
        question = ""

    # Remove "Read the following context:" prefix
    prefix = "Read the following context:"
    if context.startswith(prefix):
        context = context[len(prefix):].strip()

    # Clean trailing ".." from context
    context = re.sub(r"\.\.\s*$", ".", context)

    return clean_text(context), clean_text(question)


def format_choices(choices: dict, answer_key: str) -> str:
    """Format SimpleToM multiple-choice answer from choices dict."""
    if not choices or "text" not in choices or "label" not in choices:
        return ""

    for text, label in zip(choices["text"], choices["label"]):
        if label == answer_key:
            return text
    return ""
