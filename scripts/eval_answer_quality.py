"""Evaluate answer quality: routed vs. always-ToM vs. always-social.

Generates expert answers on a stratified test sample and compares
token F1 / exact match between three end-to-end policies:

  1. routed         — router decides per question, picks ToM or social expert
  2. always_tom     — every question goes to the ToM expert
  3. always_social  — every question goes to the social expert

The point is to verify that the router preserves answer quality vs.
always-ToM (the strongest non-router baseline) — especially on the
ToM subset, where we explicitly want no quality loss.

Both experts wrap the same OLMo-3-7B model and differ only in their
system prompt, so the cost difference between policies is the cost of
running an unnecessary expert, not the cost of loading two models.

For each sampled (context, question) pair we generate exactly two answers
(tom_answer, social_answer) and reuse them across all three policies, so
total generation cost is `2 * n_samples` regardless of policy count.

Outputs:
  - outputs/reports/answer_quality_eval.json         (aggregated metrics)
  - outputs/reports/answer_quality_predictions.parquet (per-sample answers + scores)
  - outputs/answer_quality/answer_cache.jsonl       (incremental generation cache)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.metrics_qa import exact_match, token_f1
from src.inference.router_pipeline import RouterPipeline
from src.models.experts import OLMoExpert
from src.utils.seed import set_seed


def load_olmo(model_name: str, load_in_4bit: bool = True):
    """Load OLMo-3 with the same 4-bit config used elsewhere in the project."""
    quant = None
    if load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def stratified_sample(df: pd.DataFrame, n_per_class: int, seed: int = 42) -> pd.DataFrame:
    """Take an equal-sized stratified sample of ToM and non-ToM rows."""
    tom = df[df["requires_tom"] == 1]
    non_tom = df[df["requires_tom"] == 0]
    n_tom = min(n_per_class, len(tom))
    n_non = min(n_per_class, len(non_tom))
    sample = pd.concat(
        [
            tom.sample(n=n_tom, random_state=seed),
            non_tom.sample(n=n_non, random_state=seed + 1),
        ]
    ).sample(frac=1, random_state=seed + 2).reset_index(drop=True)
    return sample


def cache_load(path: Path) -> dict:
    """Load incremental generation cache (JSONL) keyed by sample_id."""
    if not path.exists():
        return {}
    cache = {}
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                cache[rec["sample_id"]] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


def cache_append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def policy_metrics(df: pd.DataFrame, f1_col: str, em_col: str, mask=None) -> dict:
    if mask is not None:
        df = df[mask]
    if len(df) == 0:
        return {"n": 0, "token_f1": 0.0, "exact_match": 0.0}
    return {
        "n": int(len(df)),
        "token_f1": float(df[f1_col].mean()),
        "exact_match": float(df[em_col].mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/processed/router_dataset_hardened_v2.parquet",
    )
    parser.add_argument(
        "--router-path",
        default="outputs/checkpoints/v2_DeBERTa_distilled",
        help="Folder containing best_f1/model.pt",
    )
    parser.add_argument("--router-model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--olmo-model", default="allenai/Olmo-3-7B-Instruct")
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=100,
        help="Stratified sample size per ToM/non-ToM class (default: 100/100 = 200 total)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Router probability threshold (defaults to threshold_f1_best in router_eval_metrics.json)",
    )
    parser.add_argument("--output-dir", default="outputs/reports")
    parser.add_argument("--cache", default="outputs/answer_quality/answer_cache.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Max tokens to generate per answer",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of greedy decoding (default: greedy for reproducibility)",
    )
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache)

    # 1. Load test split, drop rows without a gold answer
    df = pd.read_parquet(args.dataset)
    test = df[df["split"] == "test"].copy()
    has_gold = test["answer"].notna() & (test["answer"].astype(str).str.strip() != "")
    test = test[has_gold]
    sample = stratified_sample(test, args.n_per_class, seed=args.seed)
    n_tom = int((sample["requires_tom"] == 1).sum())
    n_non = int((sample["requires_tom"] == 0).sum())
    print(
        f"Sampled {len(sample)} from {len(test)} test samples "
        f"(ToM={n_tom}, non-ToM={n_non})"
    )

    # 2. Resolve router threshold
    threshold = args.threshold
    if threshold is None:
        metrics_path = Path("outputs/reports/router_eval_metrics.json")
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
                threshold = (
                    m.get("thresholds", {})
                    .get("threshold_f1_best", {})
                    .get("threshold", 0.5)
                )
        else:
            threshold = 0.5
    print(f"Router threshold: {threshold:.3f}")

    # 3. Load router and OLMo (one model, two experts)
    router = RouterPipeline(
        model_path=str(Path(args.router_path) / "best_f1"),
        model_name=args.router_model_name,
        threshold=threshold,
    )

    print(f"Loading {args.olmo_model} (4-bit={'no' if args.no_4bit else 'yes'})...")
    t0 = time.time()
    olmo_model, olmo_tok = load_olmo(args.olmo_model, load_in_4bit=not args.no_4bit)
    expert_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
    )
    tom_expert = OLMoExpert(olmo_model, olmo_tok, role="tom", **expert_kwargs)
    social_expert = OLMoExpert(olmo_model, olmo_tok, role="social", **expert_kwargs)
    print(f"Loaded in {time.time() - t0:.1f}s")

    # 4. Generate answers (cached)
    cache = cache_load(cache_path)
    if cache:
        print(f"Loaded {len(cache)} cached answers from {cache_path}")

    rows = []
    t_start = time.time()
    for i, row in enumerate(sample.itertuples(index=False), 1):
        sid = row.sample_id
        ctx, q, gold = row.context, row.question, str(row.answer)

        if sid in cache:
            rec = dict(cache[sid])
        else:
            tom_out = tom_expert.predict(ctx, q)
            soc_out = social_expert.predict(ctx, q)
            rec = {
                "sample_id": sid,
                "tom_answer": tom_out["answer"],
                "social_answer": soc_out["answer"],
            }
            cache_append(cache_path, rec)
            cache[sid] = rec

        # Router decision
        rdec = router.predict(ctx, q)
        rec["prob_tom"] = float(rdec["prob_tom"])
        rec["router_route"] = rdec["route"]
        rec["routed_answer"] = (
            rec["tom_answer"] if rdec["route"] == "tom" else rec["social_answer"]
        )
        rec["gold_answer"] = gold
        rec["requires_tom"] = int(row.requires_tom)
        rec["source_dataset"] = str(row.source_dataset)
        rec["context"] = ctx
        rec["question"] = q
        rows.append(rec)

        if i % 25 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / i * (len(sample) - i)
            print(f"  [{i}/{len(sample)}] elapsed={elapsed:.0f}s eta={eta:.0f}s")

    pred_df = pd.DataFrame(rows)

    # 5. Score every (policy, sample) pair
    for policy_col, ans_col in [
        ("routed", "routed_answer"),
        ("always_tom", "tom_answer"),
        ("always_social", "social_answer"),
    ]:
        pred_df[f"{policy_col}_f1"] = [
            token_f1(p, r)
            for p, r in zip(pred_df[ans_col], pred_df["gold_answer"])
        ]
        pred_df[f"{policy_col}_em"] = [
            int(exact_match(p, r))
            for p, r in zip(pred_df[ans_col], pred_df["gold_answer"])
        ]

    tom_mask = pred_df["requires_tom"] == 1
    non_mask = pred_df["requires_tom"] == 0

    # Routing diagnostics
    routed_correct = (
        ((pred_df["router_route"] == "tom") & tom_mask)
        | ((pred_df["router_route"] == "social") & non_mask)
    )
    routing_diag = {
        "tom_route_count": int((pred_df["router_route"] == "tom").sum()),
        "social_route_count": int((pred_df["router_route"] == "social").sum()),
        "tom_recall": float(
            ((pred_df["router_route"] == "tom") & tom_mask).sum()
            / max(int(tom_mask.sum()), 1)
        ),
        "non_tom_recall": float(
            ((pred_df["router_route"] == "social") & non_mask).sum()
            / max(int(non_mask.sum()), 1)
        ),
        "routing_accuracy": float(routed_correct.mean()),
    }

    def all_policies(mask=None) -> dict:
        return {
            policy: policy_metrics(
                pred_df, f"{policy}_f1", f"{policy}_em", mask=mask
            )
            for policy in ("routed", "always_tom", "always_social")
        }

    results = {
        "config": {
            "n_total": int(len(pred_df)),
            "n_tom": int(tom_mask.sum()),
            "n_non_tom": int(non_mask.sum()),
            "router_threshold": float(threshold),
            "olmo_model": args.olmo_model,
            "router_path": args.router_path,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "router_diagnostics": routing_diag,
        "overall": all_policies(),
        "tom_subset": all_policies(tom_mask),
        "non_tom_subset": all_policies(non_mask),
    }

    # Per-source breakdown (only sources with >= 5 samples)
    per_source = {}
    for src, group in pred_df.groupby("source_dataset"):
        if len(group) < 5:
            continue
        src_mask = pred_df["source_dataset"] == src
        per_source[str(src)] = {
            "n": int(src_mask.sum()),
            **all_policies(src_mask),
        }
    results["per_source"] = per_source

    # 6. Print summary
    print("\n" + "=" * 80)
    print("ANSWER QUALITY: routed vs always-ToM vs always-social")
    print("=" * 80)
    print(
        f"{'Subset':<18} {'Policy':<15} {'n':>5} "
        f"{'token F1':>10} {'EM':>8}"
    )
    print("-" * 80)
    for subset_name in ("overall", "tom_subset", "non_tom_subset"):
        for policy in ("routed", "always_tom", "always_social"):
            m = results[subset_name][policy]
            print(
                f"{subset_name:<18} {policy:<15} {m['n']:>5} "
                f"{m['token_f1']:>10.4f} {m['exact_match']:>8.4f}"
            )
        print()

    # Headline comparisons
    o = results["overall"]
    t = results["tom_subset"]
    print(
        "Δ(routed - always_tom) overall  : "
        f"F1 {o['routed']['token_f1'] - o['always_tom']['token_f1']:+.4f}, "
        f"EM {o['routed']['exact_match'] - o['always_tom']['exact_match']:+.4f}"
    )
    print(
        "Δ(routed - always_tom) ToM only : "
        f"F1 {t['routed']['token_f1'] - t['always_tom']['token_f1']:+.4f}, "
        f"EM {t['routed']['exact_match'] - t['always_tom']['exact_match']:+.4f}"
    )
    print(
        f"Router routing accuracy on this sample: {routing_diag['routing_accuracy']:.4f}"
    )

    # 7. Save
    out_json = output_dir / "answer_quality_eval.json"
    out_pq = output_dir / "answer_quality_predictions.parquet"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    pred_df.to_parquet(out_pq, index=False)
    print(f"\nSaved aggregated metrics to {out_json}")
    print(f"Saved per-sample predictions to {out_pq}")


if __name__ == "__main__":
    main()
