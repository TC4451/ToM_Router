"""Data collators for router training."""

import torch
from transformers import PreTrainedTokenizer


class RouterCollator:
    """Collator for router training batches.

    Handles padding and creates tensors for input_ids, attention_mask,
    hard labels, and teacher soft labels.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch: list[dict]) -> dict:
        texts = [item["text"] for item in batch]
        hard_labels = torch.tensor([item["requires_tom"] for item in batch], dtype=torch.float)
        teacher_probs = torch.tensor([item["teacher_prob_tom"] for item in batch], dtype=torch.float)

        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "hard_labels": hard_labels,
            "teacher_probs": teacher_probs,
        }
