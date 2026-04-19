"""Router inference pipeline."""

import numpy as np
import torch
from src.models.router_student import StudentRouter, get_tokenizer


class RouterPipeline:
    """Inference pipeline for the trained student router."""

    def __init__(
        self,
        model_path: str,
        model_name: str = "microsoft/deberta-v3-base",
        threshold: float = 0.5,
        device: str = None,
    ):
        self.threshold = threshold
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = StudentRouter(model_name=model_name)
        state_dict = torch.load(
            f"{model_path}/model.pt", map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = get_tokenizer(model_name)

    @torch.no_grad()
    def predict(self, context: str, question: str) -> dict:
        """Predict whether a query requires ToM reasoning."""
        text = f"[CONTEXT] {context} [QUESTION] {question}"
        inputs = self.tokenizer(
            text, max_length=512, truncation=True, return_tensors="pt"
        ).to(self.device)

        logit = self.model(**inputs).item()
        prob = 1 / (1 + np.exp(-logit))

        return {
            "prob_tom": float(prob),
            "requires_tom": int(prob >= self.threshold),
            "route": "tom" if prob >= self.threshold else "social",
            "uncertainty": float(1 - abs(2 * prob - 1)),  # 0=certain, 1=max uncertain
        }

    @torch.no_grad()
    def predict_batch(self, contexts: list[str], questions: list[str]) -> list[dict]:
        """Predict for a batch of samples."""
        texts = [
            f"[CONTEXT] {c} [QUESTION] {q}" for c, q in zip(contexts, questions)
        ]
        inputs = self.tokenizer(
            texts, max_length=512, truncation=True, padding=True, return_tensors="pt"
        ).to(self.device)

        logits = self.model(**inputs).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))

        results = []
        for prob in probs:
            p = float(prob)
            results.append({
                "prob_tom": p,
                "requires_tom": int(p >= self.threshold),
                "route": "tom" if p >= self.threshold else "social",
                "uncertainty": float(1 - abs(2 * p - 1)),
            })
        return results
