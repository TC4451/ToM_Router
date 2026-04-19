"""Student router model — compact binary classifier for ToM routing."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class StudentRouter(nn.Module):
    """Binary classifier using a pretrained encoder (e.g., DeBERTa-v3-base).

    Input: "[CONTEXT] {context} [QUESTION] {question}"
    Output: single logit for P(requires_tom)
    """

    def __init__(self, model_name: str = "microsoft/deberta-v3-base", dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Pool: use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :].float()
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output).squeeze(-1)
        return logits


def get_tokenizer(model_name: str = "microsoft/deberta-v3-base"):
    """Get tokenizer for the student router."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def tokenize_sample(
    tokenizer,
    context: str,
    question: str,
    max_length: int = 512,
) -> dict:
    """Tokenize a single context-question pair for the router.

    Format: "[CONTEXT] {context} [QUESTION] {question}"
    Truncates context first to preserve the full question.
    """
    # Build input with special markers
    text = f"[CONTEXT] {context} [QUESTION] {question}"
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    return encoding
