"""Teacher router using OLMo-3-7B-Instruct for ToM classification."""

import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class OLMoTeacherRouter:
    """Teacher router that uses OLMo-3-7B-Instruct to classify ToM requirements."""

    def __init__(
        self,
        model_name: str = "allenai/Olmo-3-7B-Instruct",
        load_in_4bit: bool = True,
        device: str = "auto",
    ):
        self.model_name = model_name

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device,
            dtype=torch.bfloat16,
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def _build_prompt(self, context: str, question: str) -> str:
        """Build a compact classification prompt."""
        # Truncate context to save tokens/time
        ctx = context[:600] if len(context) > 600 else context

        system = (
            "Classify if answering this question needs Theory of Mind (reasoning about "
            "hidden beliefs, false beliefs, intentions, knowledge states). "
            "Reply ONLY with JSON: {\"requires_tom\": 0or1, \"prob_tom\": 0.0to1.0, \"rationale\": \"...\"}"
        )
        user = f"Context: {ctx}\nQuestion: {question}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _parse_response(self, text: str) -> dict:
        """Parse JSON from model response, with fallback extraction."""
        text = text.strip()

        # Extract JSON from markdown code blocks if present
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        # Try to find JSON object in text
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return {
                    "teacher_label": int(data.get("requires_tom", 0)),
                    "teacher_prob_tom": float(data.get("prob_tom", 0.5)),
                    "teacher_rationale": str(data.get("rationale", "")),
                }
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: regex extraction
        tom_match = re.search(r"requires_tom[\"']?\s*:\s*(\d)", text)
        prob_match = re.search(r"prob_tom[\"']?\s*:\s*([\d.]+)", text)
        rat_match = re.search(r"rationale[\"']?\s*:\s*[\"']([^\"']+)[\"']", text)

        return {
            "teacher_label": int(tom_match.group(1)) if tom_match else 0,
            "teacher_prob_tom": float(prob_match.group(1)) if prob_match else 0.5,
            "teacher_rationale": rat_match.group(1) if rat_match else "parse_failed",
        }

    @torch.no_grad()
    def predict(self, context: str, question: str) -> dict:
        """Generate teacher label for a single sample."""
        prompt = self._build_prompt(context, question)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self._parse_response(response)

    @torch.no_grad()
    def predict_batch(self, contexts: list[str], questions: list[str], batch_size: int = 4) -> list[dict]:
        """Generate teacher labels for a batch using left-padded batched generation."""
        results = []
        for i in range(0, len(contexts), batch_size):
            batch_ctx = contexts[i:i + batch_size]
            batch_q = questions[i:i + batch_size]

            prompts = [self._build_prompt(c, q) for c, q in zip(batch_ctx, batch_q)]
            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            for j, output in enumerate(outputs):
                input_len = inputs["input_ids"][j].shape[0]
                new_tokens = output[input_len:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                results.append(self._parse_response(response))

        return results
