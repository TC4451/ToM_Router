"""Style-normalize all samples into a uniform narrative format using OLMo-3.

Rewrites contexts from all source datasets into a consistent third-person
narrative style, removing source-specific vocabulary and formatting cues.
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
CACHE_PATH = Path("outputs/style_norm/style_cache.jsonl")
OUTPUT_PATH = Path("outputs/style_norm/router_dataset_normalized.parquet")

LOG_EVERY = 100


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


def build_normalize_prompt(context, question, tokenizer):
    """Build prompt to rewrite context+question in uniform style."""
    ctx_truncated = context[:600]

    user_msg = (
        "Rewrite the following social scenario and question in a uniform third-person "
        "narrative style. Keep ALL factual content, characters, events, and meaning "
        "EXACTLY the same. Only change the writing style to be a clean, neutral, "
        "third-person narrative paragraph. Do NOT add or remove any information.\n\n"
        f"Original context: {ctx_truncated}\n"
        f"Original question: {question}\n\n"
        "Return JSON: {\"context\": \"rewritten context\", \"question\": \"rewritten question\"}\n"
        "Return ONLY valid JSON."
    )

    messages = [
        {"role": "system", "content": "You rewrite text into uniform narrative style. Preserve all meaning exactly. Return ONLY valid JSON."},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_normalize_response(text):
    """Parse JSON response."""
    text = text.strip()
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            ctx = str(data.get("context", ""))
            q = str(data.get("question", ""))
            if len(ctx) > 20 and len(q) > 5:
                return {"context": ctx, "question": q}
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def main():
    set_seed(42)

    print("Loading dataset...")
    df = pd.read_parquet(DATASET_PATH)
    print(f"  {len(df)} samples")

    # Load cache
    cache = load_cache(CACHE_PATH)
    uncached = df[~df["sample_id"].isin(cache.keys())].reset_index(drop=True)
    print(f"  {len(cache)} already cached, {len(uncached)} to process")

    if len(uncached) > 0:
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
            prompt = build_normalize_prompt(row["context"], row["question"], tokenizer)

            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=180,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)

                parsed = parse_normalize_response(response)
                if parsed:
                    record = {
                        "sample_id": row["sample_id"],
                        "norm_context": parsed["context"],
                        "norm_question": parsed["question"],
                        "status": "ok",
                    }
                else:
                    record = {
                        "sample_id": row["sample_id"],
                        "norm_context": row["context"],
                        "norm_question": row["question"],
                        "status": "parse_failed_kept_original",
                    }
            except Exception as e:
                record = {
                    "sample_id": row["sample_id"],
                    "norm_context": row["context"],
                    "norm_question": row["question"],
                    "status": f"error_kept_original: {str(e)[:60]}",
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
                    f"{ok_count}/{processed} ok"
                )

        elapsed = time.time() - start_time
        print(f"\nProcessed {processed} in {elapsed/60:.1f}min ({processed/elapsed:.2f}/sec)")

        del model
        torch.cuda.empty_cache()

    # Build normalized dataset
    print("\nBuilding normalized dataset...")
    ok_count = sum(1 for v in cache.values() if v.get("status") == "ok")
    fail_count = len(cache) - ok_count
    print(f"  {ok_count} successfully normalized, {fail_count} kept original")

    norm_df = pd.DataFrame(list(cache.values()))
    result = df.merge(norm_df[["sample_id", "norm_context", "norm_question", "status"]],
                      on="sample_id", how="left")

    # Replace context/question with normalized versions
    has_norm = result["norm_context"].notna()
    result.loc[has_norm, "context"] = result.loc[has_norm, "norm_context"]
    result.loc[has_norm, "question"] = result.loc[has_norm, "norm_question"]
    result = result.drop(columns=["norm_context", "norm_question", "status"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)
    print(f"  Saved {len(result)} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
