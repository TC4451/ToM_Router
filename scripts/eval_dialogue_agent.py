"""Evaluate the dialogue agent across three reasoning policies.

Each turn provides a (context, question) pair from the test set.
The router classifies each pair directly (same format as training).
Conversation history affects expert response quality, not routing decisions.

This measures: in a stream of mixed social reasoning questions,
can the adaptive router save cost by correctly identifying which
questions need expensive ToM reasoning?
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.seed import set_seed
from src.inference.router_pipeline import RouterPipeline
from src.eval.metrics_qa import token_f1

sns.set_theme(style="whitegrid", font_scale=1.1)
FIG_DIR = Path("outputs/figures")
OUT_DIR = Path("outputs/reports")
SCENARIOS_PATH = Path("data/processed/dialogue_scenarios.json")

# Realistic cost model: ToM reasoning uses longer system prompts,
# chain-of-thought, and generates more detailed responses
TOM_PROMPT_OVERHEAD = 200   # extra tokens for ToM system prompt + CoT instructions
SOCIAL_PROMPT_OVERHEAD = 50  # lighter social prompt
TOM_OUTPUT_MULTIPLIER = 2.5  # ToM responses are ~2.5x longer (reasoning traces)
SOCIAL_OUTPUT_MULTIPLIER = 1.0


def simulate_expert_cost(context: str, question: str, route: str) -> dict:
    """Simulate realistic token costs for each expert."""
    input_tokens = len(context.split()) + len(question.split())

    if route == "tom":
        total_in = input_tokens + TOM_PROMPT_OVERHEAD
        base_out = max(30, int(len(question.split()) * 3))
        total_out = int(base_out * TOM_OUTPUT_MULTIPLIER)
    else:
        total_in = input_tokens + SOCIAL_PROMPT_OVERHEAD
        base_out = max(15, int(len(question.split()) * 2))
        total_out = int(base_out * SOCIAL_OUTPUT_MULTIPLIER)

    return {"tokens_in": total_in, "tokens_out": total_out, "total_tokens": total_in + total_out}


def run_policy(policy_name: str, scenarios: list, router=None) -> dict:
    """Run all scenarios under one policy."""
    all_turns = []
    per_scenario = {}

    for scenario in scenarios:
        turns = []
        for i, turn in enumerate(scenario["turns"]):
            context = turn["context"]
            question = turn["question"]
            gold = turn["requires_tom"]
            expected = turn["expected_route"]

            # Decide route
            start = time.time()
            if policy_name == "always_tom":
                route = "tom"
            elif policy_name == "general_social":
                route = "social"
            else:  # adaptive
                result = router.predict(context, question)
                route = result["route"]
            latency = (time.time() - start) * 1000

            # Simulate cost
            cost = simulate_expert_cost(context, question, route)
            correct = (route == expected)

            turn_result = {
                "turn_id": i,
                "route": route,
                "expected_route": expected,
                "correct_route": correct,
                "requires_tom": gold,
                "tokens_total": cost["total_tokens"],
                "tokens_in": cost["tokens_in"],
                "tokens_out": cost["tokens_out"],
                "latency_ms": latency,
                "source": turn.get("source", ""),
            }
            turns.append(turn_result)

        # Per-scenario metrics
        n = len(turns)
        route_acc = sum(t["correct_route"] for t in turns) / n
        total_tokens = sum(t["tokens_total"] for t in turns)
        tom_count = sum(1 for t in turns if t["route"] == "tom")

        per_scenario[scenario["scenario_id"]] = {
            "type": scenario["type"],
            "transition_turn": scenario.get("transition_turn"),
            "n_turns": n,
            "routing_accuracy": route_acc,
            "total_tokens": total_tokens,
            "avg_tokens_per_turn": total_tokens / n,
            "tom_ratio": tom_count / n,
            "routes": [t["route"] for t in turns],
            "expected_routes": [t["expected_route"] for t in turns],
            "correct_routes": [t["correct_route"] for t in turns],
        }
        all_turns.extend(turns)

    # Aggregate
    total_correct = sum(t["correct_route"] for t in all_turns)
    total_n = len(all_turns)
    total_tokens = sum(t["tokens_total"] for t in all_turns)
    total_tom = sum(1 for t in all_turns if t["route"] == "tom")

    # Per-type breakdown
    type_metrics = {}
    for stype in set(s["type"] for s in scenarios):
        type_turns = [t for s in scenarios for i, t in enumerate(
            [tt for tt in all_turns if any(
                tt["turn_id"] == j and per_scenario[s["scenario_id"]]["type"] == stype
                for j in range(len(s["turns"]))
            )]
        )]
        # Simpler approach
        type_scenarios = {s_id: data for s_id, data in per_scenario.items() if data["type"] == stype}
        if type_scenarios:
            type_metrics[stype] = {
                "routing_accuracy": np.mean([d["routing_accuracy"] for d in type_scenarios.values()]),
                "avg_tokens_per_turn": np.mean([d["avg_tokens_per_turn"] for d in type_scenarios.values()]),
                "tom_ratio": np.mean([d["tom_ratio"] for d in type_scenarios.values()]),
                "n_scenarios": len(type_scenarios),
            }

    return {
        "policy": policy_name,
        "routing_accuracy": total_correct / total_n,
        "total_tokens": total_tokens,
        "avg_tokens_per_turn": total_tokens / total_n,
        "tom_ratio": total_tom / total_n,
        "n_turns": total_n,
        "per_type": type_metrics,
        "per_scenario": per_scenario,
    }


def make_figures(results: dict):
    """Generate all comparison figures."""
    policies = list(results.keys())
    policy_colors = {"always_tom": "#E74C3C", "general_social": "#3498DB", "adaptive_router": "#2ECC71"}
    policy_labels = {"always_tom": "Always ToM", "general_social": "General Social", "adaptive_router": "Adaptive Router"}

    # Fig 17: Routing accuracy by scenario type
    scenario_types = ["pure_tom", "pure_social", "mixed", "transition_social_to_tom", "transition_tom_to_social"]
    type_labels = ["Pure\nToM", "Pure\nSocial", "Mixed", "Social\n→ ToM", "ToM\n→ Social"]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(scenario_types))
    w = 0.25
    for i, policy in enumerate(policies):
        accs = [results[policy]["per_type"].get(st, {}).get("routing_accuracy", 0) for st in scenario_types]
        bars = ax.bar(x + (i - 1) * w, [a * 100 for a in accs], w,
                      label=policy_labels[policy], color=policy_colors[policy], edgecolor="white")
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{acc*100:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel("Routing Accuracy (%)")
    ax.set_title("Routing Accuracy by Policy and Scenario Type")
    ax.set_xticks(x)
    ax.set_xticklabels(type_labels)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 115)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3)
    ax.text(4.5, 52, "random chance", fontsize=8, color="gray", alpha=0.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig17_policy_routing_accuracy.png", dpi=150)
    plt.close()

    # Fig 18: Cost comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    costs = [results[p]["avg_tokens_per_turn"] for p in policies]
    accs = [results[p]["routing_accuracy"] * 100 for p in policies]
    bars = ax.bar([policy_labels[p] for p in policies], costs,
                  color=[policy_colors[p] for p in policies], edgecolor="white", width=0.5)
    for bar, cost, acc in zip(bars, costs, accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f'{cost:.0f} tokens\n({acc:.0f}% acc)', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel("Average Tokens per Turn")
    ax.set_title("Cost vs. Routing Accuracy")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig18_policy_cost.png", dpi=150)
    plt.close()

    # Fig 19: Adaptation trace for transition scenarios
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, trans_type in enumerate(["transition_social_to_tom", "transition_tom_to_social"]):
        ax = axes[ax_idx]
        # Find first scenario of this type for adaptive router
        for s_id, data in results["adaptive_router"]["per_scenario"].items():
            if data["type"] == trans_type:
                n = data["n_turns"]
                x_turns = range(1, n + 1)
                gold = [1 if r == "tom" else 0 for r in data["expected_routes"]]
                pred = [1 if r == "tom" else 0 for r in data["routes"]]

                ax.step(x_turns, gold, where="mid", label="Ground truth",
                       color="#7F8C8D", linewidth=2.5, linestyle="--")
                ax.step(x_turns, pred, where="mid", label="Adaptive router",
                       color="#2ECC71", linewidth=2.5)

                trans = data.get("transition_turn")
                if trans:
                    ax.axvline(x=trans + 0.5, color="#E74C3C", linestyle=":", alpha=0.8, linewidth=2)
                    ax.text(trans + 0.6, 0.5, "shift", rotation=90,
                           va="center", fontsize=10, color="#E74C3C", fontweight="bold")

                # Mark correct/incorrect
                for j, (g, p) in enumerate(zip(gold, pred)):
                    marker = "o" if g == p else "x"
                    color = "#2ECC71" if g == p else "#E74C3C"
                    ax.plot(j + 1, p, marker, color=color, markersize=8, zorder=5)

                ax.set_xlabel("Turn")
                ax.set_ylabel("Route")
                ax.set_yticks([0, 1])
                ax.set_yticklabels(["Social", "ToM"])
                title = "Social → ToM Transition" if "s2t" in s_id else "ToM → Social Transition"
                ax.set_title(title)
                ax.legend(loc="center right", fontsize=9)
                for spine in ["top", "right"]:
                    ax.spines[spine].set_visible(False)
                break

    plt.suptitle("Adaptation: How the Router Responds to Mid-Conversation Shifts",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig19_adaptation_trace.png", dpi=150)
    plt.close()

    # Fig 20: Cost-quality Pareto
    fig, ax = plt.subplots(figsize=(8, 6))
    for policy in policies:
        per_s = results[policy]["per_scenario"]
        costs_s = [d["avg_tokens_per_turn"] for d in per_s.values()]
        accs_s = [d["routing_accuracy"] * 100 for d in per_s.values()]
        ax.scatter(costs_s, accs_s, label=policy_labels[policy],
                  color=policy_colors[policy], s=60, alpha=0.6, edgecolor="white")
        # Mean point
        ax.scatter([np.mean(costs_s)], [np.mean(accs_s)],
                  color=policy_colors[policy], s=200, edgecolor="black", linewidth=2, zorder=5,
                  marker="D")

    ax.set_xlabel("Average Tokens per Turn (cost →)")
    ax.set_ylabel("Routing Accuracy (%) (quality →)")
    ax.set_title("Cost-Quality Tradeoff (diamonds = policy mean)")
    ax.legend()
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig20_cost_quality_pareto.png", dpi=150)
    plt.close()

    print("  fig17–fig20 generated")


def main():
    set_seed(42)

    print("Loading scenarios...")
    with open(SCENARIOS_PATH) as f:
        scenarios = json.load(f)
    print(f"  {len(scenarios)} scenarios, {sum(len(s['turns']) for s in scenarios)} total turns")

    # Load router
    print("Loading router...")
    router = RouterPipeline(
        model_path="outputs/checkpoints/router_student/best_f1",
        model_name="microsoft/deberta-v3-base",
        threshold=0.72,
    )

    # Run all policies
    results = {}
    for policy in ["always_tom", "general_social", "adaptive_router"]:
        print(f"\nRunning: {policy}...")
        r = run_policy(policy, scenarios, router=router if policy == "adaptive_router" else None)
        results[policy] = r
        print(f"  routing_accuracy={r['routing_accuracy']:.4f}, "
              f"avg_tokens/turn={r['avg_tokens_per_turn']:.0f}, "
              f"tom_ratio={r['tom_ratio']:.2f}")

    # Summary table
    print("\n" + "=" * 80)
    print("DOWNSTREAM AGENT COMPARISON")
    print("=" * 80)

    # Compute cost relative to always-tom
    tom_cost = results["always_tom"]["avg_tokens_per_turn"]
    tom_acc = results["always_tom"]["routing_accuracy"]

    print(f"\n{'Policy':<20} {'Route Acc':>10} {'Tokens/Turn':>12} {'Cost vs ToM':>12} {'ToM Usage':>10}")
    print("-" * 66)
    for policy in ["always_tom", "general_social", "adaptive_router"]:
        r = results[policy]
        cost_ratio = r["avg_tokens_per_turn"] / tom_cost
        print(f"{policy:<20} {r['routing_accuracy']*100:>9.1f}% "
              f"{r['avg_tokens_per_turn']:>12.0f} {cost_ratio:>11.0%} "
              f"{r['tom_ratio']*100:>9.0f}%")

    # By scenario type
    print(f"\n{'Type':<25} {'Always ToM':>12} {'Gen Social':>12} {'Adaptive':>12}")
    print("-" * 64)
    for stype in ["pure_tom", "pure_social", "mixed", "transition_social_to_tom", "transition_tom_to_social"]:
        vals = []
        for policy in ["always_tom", "general_social", "adaptive_router"]:
            acc = results[policy]["per_type"].get(stype, {}).get("routing_accuracy", 0)
            vals.append(f"{acc*100:.0f}%")
        print(f"{stype:<25} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    # Cost savings analysis
    adaptive = results["adaptive_router"]
    print(f"\n--- Cost Savings Analysis ---")
    print(f"  Always-ToM cost:     {results['always_tom']['avg_tokens_per_turn']:.0f} tokens/turn")
    print(f"  Adaptive cost:       {adaptive['avg_tokens_per_turn']:.0f} tokens/turn")
    print(f"  Savings:             {(1 - adaptive['avg_tokens_per_turn']/tom_cost)*100:.0f}%")
    print(f"  Routing accuracy:    {adaptive['routing_accuracy']*100:.1f}%")
    print(f"  ToM expert usage:    {adaptive['tom_ratio']*100:.0f}% of turns (vs 100% for always-ToM)")

    # Generate figures
    print("\nGenerating figures...")
    make_figures(results)

    # Save
    save_data = {
        "summary": [
            {"policy": p, "routing_accuracy": results[p]["routing_accuracy"],
             "avg_tokens_per_turn": results[p]["avg_tokens_per_turn"],
             "tom_ratio": results[p]["tom_ratio"],
             "cost_ratio": results[p]["avg_tokens_per_turn"] / tom_cost}
            for p in results
        ],
        "per_type": {p: results[p]["per_type"] for p in results},
        "per_scenario": {p: {
            s_id: {k: v for k, v in data.items() if k != "turns"}
            for s_id, data in results[p]["per_scenario"].items()
        } for p in results},
    }
    with open(OUT_DIR / "dialogue_agent_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nResults saved to {OUT_DIR / 'dialogue_agent_results.json'}")


if __name__ == "__main__":
    main()
