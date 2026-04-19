"""Generate contrastive question pairs using OLMo-3.

For each ToM context, generate a non-ToM question about the same context.
For each non-ToM context, generate a ToM question about the same context.
This ensures both labels exist for the same contexts, breaking source shortcuts.
"""

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.seed import set_seed

DATASET_PATH = Path("data/processed/router_dataset.parquet")
CACHE_PATH = Path("outputs/contrastive/contrastive_cache.jsonl")
OUTPUT_PATH = Path("outputs/contrastive/contrastive_questions.parquet")

LOG_EVERY = 50


def load_cache(path):
    cache = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    cache[rec["sample_id"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue
    return cache


def append_cache(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def build_contrastive_prompt(context, original_question, original_label, tokenizer):
    """Build prompt to generate a contrastive question."""
    if original_label == 1:
        # Original is ToM -> generate non-ToM question
        task = (
            "Given this context and a Theory-of-Mind question about it, "
            "write a NEW question about the SAME context that does NOT require "
            "Theory of Mind. The new question should be answerable from "
            "observable facts, emotions, social norms, or relationships — "
            "without needing to reason about hidden beliefs or knowledge states.\n\n"
            "Return JSON: {\"question\": \"...\", \"answer\": \"...\", \"subtype\": \"emotion|norm|relation|social_commonsense\"}"
        )
    else:
        # Original is non-ToM -> generate ToM question
        task = (
            "Given this context and a social reasoning question about it, "
            "write a NEW question about the SAME context that REQUIRES "
            "Theory of Mind reasoning. The new question should require "
            "reasoning about a character's hidden beliefs, false beliefs, "
            "intentions, knowledge state, or perspective.\n\n"
            "Return JSON: {\"question\": \"...\", \"answer\": \"...\", \"subtype\": \"belief|intention|desire\"}"
        )

    ctx_truncated = context[:500]
    user_msg = (
        f"{task}\n\n"
        f"Context: {ctx_truncated}\n"
        f"Original question (for reference): {original_question}\n\n"
        f"Return ONLY valid JSON."
    )

    messages = [
        {"role": "system", "content": "You generate contrastive questions for social reasoning research. Return ONLY valid JSON."},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_contrastive_response(text):
    """Parse JSON response for contrastive question."""
    text = text.strip()
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return {
                "question": str(data.get("question", "")),
                "answer": str(data.get("answer", "")),
                "subtype": str(data.get("subtype", "other")),
            }
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def main():
    set_seed(42)

    print("Loading dataset...")
    df = pd.read_parquet(DATASET_PATH)
    print(f"  {len(df)} samples")

    # Sample a balanced subset for contrastive generation
    # We don't need contrastive for ALL samples — a good subset suffices
    tom_samples = df[df["requires_tom"] == 1].sample(n=min(1500, len(df[df["requires_tom"] == 1])), random_state=42)
    non_tom_samples = df[df["requires_tom"] == 0].sample(n=min(1500, len(df[df["requires_tom"] == 0])), random_state=42)
    samples = pd.concat([tom_samples, non_tom_samples]).reset_index(drop=True)
    print(f"  Generating contrastive questions for {len(samples)} samples")

    # Load cache
    cache = load_cache(CACHE_PATH)
    uncached = samples[~samples["sample_id"].isin(cache.keys())].reset_index(drop=True)
    print(f"  {len(cache)} already cached, {len(uncached)} to process")

    if len(uncached) > 0:
        # Load model
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print("\nLoading OLMo-3-7B-Instruct...")
        tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            "allenai/Olmo-3-7B-Instruct",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            ),
            device_map="auto",
            dtype=torch.bfloat16,
        )
        model.eval()
        print("  Model loaded")

        start_time = time.time()
        processed = 0

        for idx, row in uncached.iterrows():
            prompt = build_contrastive_prompt(
                row["context"], row["question"], row["requires_tom"], tokenizer
            )

            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=120,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)

                parsed = parse_contrastive_response(response)
                if parsed and len(parsed["question"]) > 10:
                    new_label = 0 if row["requires_tom"] == 1 else 1
                    record = {
                        "sample_id": row["sample_id"],
                        "original_label": int(row["requires_tom"]),
                        "contrastive_question": parsed["question"],
                        "contrastive_answer": parsed["answer"],
                        "contrastive_label": new_label,
                        "contrastive_subtype": parsed["subtype"],
                        "status": "ok",
                    }
                else:
                    record = {
                        "sample_id": row["sample_id"],
                        "original_label": int(row["requires_tom"]),
                        "contrastive_question": "",
                        "contrastive_answer": "",
                        "contrastive_label": -1,
                        "contrastive_subtype": "",
                        "status": "parse_failed",
                    }
            except Exception as e:
                record = {
                    "sample_id": row["sample_id"],
                    "original_label": int(row["requires_tom"]),
                    "contrastive_question": "",
                    "contrastive_answer": "",
                    "contrastive_label": -1,
                    "contrastive_subtype": "",
                    "status": f"error: {str(e)[:80]}",
                }

            cache[row["sample_id"]] = record
            append_cache(CACHE_PATH, [record])
            processed += 1

            if processed % LOG_EVERY == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                remaining = (len(uncached) - processed) / rate
                ok_count = sum(1 for v in cache.values() if v.get("status") == "ok")
                print(
                    f"  [{processed}/{len(uncached)}] {rate:.2f}/sec, "
                    f"~{remaining/60:.0f}min left, "
                    f"{ok_count} ok so far"
                )

        elapsed = time.time() - start_time
        print(f"\nProcessed {processed} in {elapsed/60:.1f}min ({processed/elapsed:.2f}/sec)")

        del model
        torch.cuda.empty_cache()

    # Build contrastive dataset
    print("\nBuilding contrastive dataset...")
    ok_records = [v for v in cache.values() if v.get("status") == "ok"]
    print(f"  {len(ok_records)} successful contrastive questions")

    # Merge with original contexts
    contrastive_df = pd.DataFrame(ok_records)
    original_df = df[["sample_id", "context", "source_dataset", "split"]].copy()
    merged = contrastive_df.merge(original_df, on="sample_id", how="inner")

    # Build new samples
    new_rows = []
    for _, row in merged.iterrows():
        new_rows.append({
            "sample_id": f"contrastive_{row['sample_id']}",
            "source_dataset": f"{row['source_dataset']}_contrastive",
            "context": row["context"],
            "question": row["contrastive_question"],
            "answer": row["contrastive_answer"],
            "requires_tom": row["contrastive_label"],
            "subtype": row["contrastive_subtype"],
            "original_category": "contrastive",
            "split": row["split"],
            "teacher_prob_tom": float(row["contrastive_label"]),  # placeholder
            "teacher_label": row["contrastive_label"],
            "teacher_rationale": "contrastive_generated",
            "metadata": {"original_sample_id": row["sample_id"]},
        })

    contrastive_out = pd.DataFrame(new_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    contrastive_out.to_parquet(OUTPUT_PATH, index=False)

    print(f"  Saved {len(contrastive_out)} contrastive samples to {OUTPUT_PATH}")
    print(f"  Label distribution: {contrastive_out['requires_tom'].value_counts().to_dict()}")
    print(f"  Source distribution: {contrastive_out['source_dataset'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
