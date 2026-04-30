"""Expert models for ToM and social reasoning."""

from abc import ABC, abstractmethod


class BaseExpert(ABC):
    """Common interface for expert models."""

    @abstractmethod
    def predict(self, context: str, question: str) -> dict:
        """Predict answer for a context-question pair.

        Returns:
            dict with keys: answer, confidence, metadata
        """
        ...


class PlaceholderToMExpert(BaseExpert):
    """Placeholder ToM expert using the teacher model with a ToM-focused prompt."""

    def __init__(self, model=None):
        self.model = model
        self.name = "tom_expert"

    def predict(self, context: str, question: str) -> dict:
        if self.model is None:
            return {
                "answer": "requires_tom_expert_model",
                "confidence": 0.0,
                "metadata": {"expert": self.name, "status": "placeholder"},
            }

        # Use model for actual prediction
        prompt = (
            f"You are an expert at Theory of Mind reasoning. "
            f"Answer the question by carefully reasoning about the characters' "
            f"hidden beliefs, knowledge states, and perspectives.\n\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Answer concisely:"
        )
        return self._generate(prompt)

    def _generate(self, prompt: str) -> dict:
        return {
            "answer": "placeholder",
            "confidence": 0.0,
            "metadata": {"expert": self.name},
        }


class PlaceholderSocialExpert(BaseExpert):
    """Placeholder social reasoning expert."""

    def __init__(self, model=None):
        self.model = model
        self.name = "social_expert"

    def predict(self, context: str, question: str) -> dict:
        if self.model is None:
            return {
                "answer": "requires_social_expert_model",
                "confidence": 0.0,
                "metadata": {"expert": self.name, "status": "placeholder"},
            }

        prompt = (
            f"You are an expert at social reasoning. "
            f"Answer the question based on social norms, emotional cues, "
            f"and observable relationship dynamics.\n\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Answer concisely:"
        )
        return self._generate(prompt)

    def _generate(self, prompt: str) -> dict:
        return {
            "answer": "placeholder",
            "confidence": 0.0,
            "metadata": {"expert": self.name},
        }


class OLMoExpert(BaseExpert):
    """Expert using OLMo-3 with role-specific prompting."""

    def __init__(
        self,
        model,
        tokenizer,
        role: str = "tom",
        max_new_tokens: int = 100,
        temperature: float = 0.3,
        do_sample: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.role = role
        self.name = f"{role}_expert"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        if role == "tom":
            self.system_prompt = (
                "You are an expert at Theory of Mind reasoning. "
                "Focus on characters' hidden beliefs, false beliefs, knowledge states, "
                "intentions, and perspectives that are not directly observable."
            )
        else:
            self.system_prompt = (
                "You are an expert at social reasoning. "
                "Focus on social norms, emotional reactions, relationship dynamics, "
                "and observable social behaviors."
            )

    def predict(self, context: str, question: str) -> dict:
        import torch

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}\nAnswer concisely:"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "answer": answer,
            "confidence": 0.8,
            "metadata": {"expert": self.name},
        }
