"""Evaluate the dialogue agent across three reasoning policies.

Compares:
1. Always-ToM: every turn uses deep ToM reasoning (expensive)
2. General-Social: every turn uses surface social reasoning (cheap)
3. Adaptive-Router: trained router decides per turn (best tradeoff)

Produces per-scenario metrics, aggregate comparisons, and figures.
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
from src.inference.dialogue_agent import DialogueAgent
from src.inference.policies import AlwaysToMPolicy, GeneralSocialPolicy, AdaptiveRouterPolicy
from src.inference.router_pipeline import RouterPipeline
from src.models.experts import PlaceholderToMExpert, PlaceholderSocialExpert
from src.eval.metrics_dialogue import compute_conversation_metrics

sns.set_theme(style="whitegrid", font_scale=1.1)
FIG_DIR = Path("outputs/figures")
OUT_DIR = Path("outputs/reports")
SCENARIOS_PATH = Path("data/processed/dialogue_scenarios.json")

# Cost multipliers: ToM expert uses longer prompts and generates more tokens
TOM_COST_MULTIPLIER = 3.0  # ToM reasoning is ~3x more expensive
SOCIAL_COST_MULTIPLIER = 1.0


class CostTrackingToMExpert(PlaceholderToMExpert):
    """ToM expert that simulates realistic token costs."""

    def predict(self, context: str, question: str) -> dict:
        tokens_in = int((len(context.split()) + len(question.split())) * TOM_COST_MULTIPLIER)
        # ToM reasoning generates longer, more detailed responses
        base_answer = f"Based on the character's perspective and hidden beliefs: analyzing mental states in this scenario."
        tokens_out = int(len(base_answer.split()) * TOM_COST_MULTIPLIER)
        return {
            "answer": base_answer,
            "confidence": 0.85,
            "metadata": {
                "expert": "tom", "tokens_in": tokens_in, "tokens_out": tokens_out,
            },
        }


class CostTrackingSocialExpert(PlaceholderSocialExpert):
    """Social expert that simulates realistic token costs."""

    def predict(self, context: str, question: str) -> dict:
        tokens_in = int((len(context.split()) + len(question.split())) * SOCIAL_COST_MULTIPLIER)
        base_answer = f"Based on social norms and observable behavior in this situation."
        tokens_out = int(len(base_answer.split()) * SOCIAL_COST_MULTIPLIER)
        return {
            "answer": base_answer,
            "confidence": 0.75,
            "metadata": {
                "expert": "social", "tokens_in": tokens_in, "tokens_out": tokens_out,
            },
        }


def run_scenario(agent, scenario):
    """Run one scenario through an agent, return per-turn results."""
    agent.reset()
    turn_results = []
    for turn in scenario["turns"]:
        result = agent.respond(
            user_message=turn["user_message"],
            gold_label=turn["requires_tom"],
        )
        result["gold_answer"] = turn.get("gold_answer", "")
        result["expected_route"] = turn["expected_route"]
        turn_results.append(result)
    return turn_results


def make_figures(all_results):
    """Generate comparison figures."""

    # Aggregate by policy and scenario type
    records = []
    for policy_name, scenarios in all_results.items():
        for s_id, data in scenarios.items():
            records.append({
                "policy": policy_name,
                "scenario_type": data["scenario_type"],
                "routing_accuracy": data["metrics"]["routing_accuracy"],
                "total_tokens": data["metrics"]["total_tokens"],
                "avg_tokens_per_turn": data["metrics"]["avg_tokens_per_turn"],
                "cost_quality_ratio": data["metrics"]["cost_quality_ratio"],
                "n_tom_routes": data["metrics"]["n_tom_routes"],
                "n_social_routes": data["metrics"]["n_social_routes"],
            })
    df = pd.DataFrame(records)

    policy_colors = {
        "always_tom": "#E74C3C",
        "general_social": "#3498DB",
        "adaptive_router": "#2ECC71",
    }
    policy_labels = {
        "always_tom": "Always ToM",
        "general_social": "General Social",
        "adaptive_router": "Adaptive Router",
    }

    # Fig 17: Routing accuracy by policy and scenario type
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot = df.pivot_table(index="scenario_type", columns="policy", values="routing_accuracy", aggfunc="mean")
    pivot = pivot.rename(columns=policy_labels)
    pivot.plot.bar(ax=ax, color=[policy_colors[p] for p in pivot.columns.map(
        {v: k for k, v in policy_labels.items()}
    )], edgecolor="white", width=0.7)
    ax.set_ylabel("Routing Accuracy")
    ax.set_title("Routing Accuracy by Policy and Scenario Type")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(title="Policy")
    ax.set_ylim(0, 1.1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig17_policy_routing_accuracy.png", dpi=150)
    plt.close()
    print("  fig17_policy_routing_accuracy.png")

    # Fig 18: Token cost by policy
    fig, ax = plt.subplots(figsize=(8, 5))
    cost_by_policy = df.groupby("policy")["avg_tokens_per_turn"].mean()
    cost_by_policy = cost_by_policy.rename(index=policy_labels)
    bars = cost_by_policy.plot.bar(ax=ax,
        color=[policy_colors[p] for p in cost_by_policy.index.map({v: k for k, v in policy_labels.items()})],
        edgecolor="white", width=0.5)
    ax.set_ylabel("Average Tokens per Turn")
    ax.set_title("Cost Comparison: Average Tokens per Turn")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig18_policy_cost.png", dpi=150)
    plt.close()
    print("  fig18_policy_cost.png")

    # Fig 19: Adaptation trace — pick one transition scenario
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, trans_type in enumerate(["transition_social_to_tom", "transition_tom_to_social"]):
        ax = axes[ax_idx]
        # Find first scenario of this type
        for s_id, data in all_results["adaptive_router"].items():
            if data["scenario_type"] == trans_type:
                turns = data["turn_results"]
                n = len(turns)
                x = range(1, n + 1)
                gold_routes = [1 if t["expected_route"] == "tom" else 0 for t in turns]
                pred_routes = [1 if t["route"] == "tom" else 0 for t in turns]

                ax.step(x, gold_routes, where="mid", label="Ground truth",
                       color="#7F8C8D", linewidth=2, linestyle="--")
                ax.step(x, pred_routes, where="mid", label="Adaptive router",
                       color="#2ECC71", linewidth=2)

                transition = data.get("transition_turn")
                if transition:
                    ax.axvline(x=transition + 0.5, color="red", linestyle=":", alpha=0.7)
                    ax.text(transition + 0.5, 0.5, "transition", rotation=90,
                           va="center", ha="right", fontsize=9, color="red", alpha=0.7)

                ax.set_xlabel("Turn")
                ax.set_ylabel("Route (1=ToM, 0=Social)")
                ax.set_yticks([0, 1])
                ax.set_yticklabels(["Social", "ToM"])
                title = "Social → ToM" if "s2t" in s_id else "ToM → Social"
                ax.set_title(f"Adaptation Trace: {title}")
                ax.legend(loc="center right")
                for spine in ["top", "right"]:
                    ax.spines[spine].set_visible(False)
                break

    plt.suptitle("How Quickly Does the Adaptive Router Adapt to Conversation Shifts?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig19_adaptation_trace.png", dpi=150)
    plt.close()
    print("  fig19_adaptation_trace.png")

    # Fig 20: Cost vs Quality scatter (per scenario)
    fig, ax = plt.subplots(figsize=(8, 6))
    for policy in ["always_tom", "general_social", "adaptive_router"]:
        subset = df[df["policy"] == policy]
        ax.scatter(subset["avg_tokens_per_turn"], subset["routing_accuracy"],
                  label=policy_labels[policy], color=policy_colors[policy],
                  s=80, alpha=0.7, edgecolor="white")
    ax.set_xlabel("Average Tokens per Turn (cost)")
    ax.set_ylabel("Routing Accuracy (quality)")
    ax.set_title("Cost-Quality Tradeoff by Policy")
    ax.legend()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig20_cost_quality_pareto.png", dpi=150)
    plt.close()
    print("  fig20_cost_quality_pareto.png")


def main():
    set_seed(42)

    print("Loading scenarios...")
    with open(SCENARIOS_PATH) as f:
        scenarios = json.load(f)
    print(f"  {len(scenarios)} scenarios loaded")

    # Build experts
    tom_expert = CostTrackingToMExpert()
    social_expert = CostTrackingSocialExpert()

    # Build router
    print("Loading router...")
    router = RouterPipeline(
        model_path="outputs/checkpoints/router_student/best_f1",
        model_name="microsoft/deberta-v3-base",
        threshold=0.72,
    )

    # Build policies
    policies = {
        "always_tom": AlwaysToMPolicy(),
        "general_social": GeneralSocialPolicy(),
        "adaptive_router": AdaptiveRouterPolicy(router),
    }

    # Run all scenarios under all policies
    all_results = {p: {} for p in policies}
    for policy_name, policy in policies.items():
        print(f"\n=== Policy: {policy_name} ===")
        agent = DialogueAgent(policy, tom_expert, social_expert, history_window=5)

        for scenario in scenarios:
            turn_results = run_scenario(agent, scenario)
            metrics = compute_conversation_metrics(turn_results, scenario)

            all_results[policy_name][scenario["scenario_id"]] = {
                "scenario_type": scenario["type"],
                "transition_turn": scenario.get("transition_turn"),
                "metrics": metrics,
                "turn_results": turn_results,
            }

        # Print aggregate
        all_metrics = [data["metrics"] for data in all_results[policy_name].values()]
        avg_acc = np.mean([m["routing_accuracy"] for m in all_metrics])
        avg_tokens = np.mean([m["avg_tokens_per_turn"] for m in all_metrics])
        avg_latency = np.mean([m["avg_latency_ms"] for m in all_metrics])
        print(f"  Avg routing accuracy: {avg_acc:.4f}")
        print(f"  Avg tokens/turn: {avg_tokens:.0f}")
        print(f"  Avg latency/turn: {avg_latency:.1f}ms")

    # Aggregate summary table
    print("\n" + "=" * 70)
    print("POLICY COMPARISON SUMMARY")
    print("=" * 70)

    summary = []
    for policy_name in policies:
        all_m = [data["metrics"] for data in all_results[policy_name].values()]
        s = {
            "policy": policy_name,
            "routing_accuracy": np.mean([m["routing_accuracy"] for m in all_m]),
            "avg_tokens_per_turn": np.mean([m["avg_tokens_per_turn"] for m in all_m]),
            "avg_latency_ms": np.mean([m["avg_latency_ms"] for m in all_m]),
            "tom_ratio": np.mean([m["n_tom_routes"] / m["n_turns"] for m in all_m]),
        }

        # Cost relative to always-tom
        summary.append(s)

    # Normalize cost relative to always-tom
    tom_cost = next(s["avg_tokens_per_turn"] for s in summary if s["policy"] == "always_tom")
    for s in summary:
        s["cost_ratio"] = s["avg_tokens_per_turn"] / tom_cost

    print(f"\n{'Policy':<20} {'Route Acc':>10} {'Tokens/Turn':>12} {'Cost Ratio':>11} {'ToM %':>7} {'Latency':>10}")
    print("-" * 72)
    for s in summary:
        print(f"{s['policy']:<20} {s['routing_accuracy']:>10.4f} "
              f"{s['avg_tokens_per_turn']:>12.0f} {s['cost_ratio']:>11.2f}x "
              f"{s['tom_ratio']:>6.1%} {s['avg_latency_ms']:>9.1f}ms")

    # By scenario type
    print("\n--- By Scenario Type ---")
    for stype in ["pure_tom", "pure_social", "mixed", "transition_social_to_tom", "transition_tom_to_social"]:
        print(f"\n  {stype}:")
        for policy_name in policies:
            type_metrics = [data["metrics"] for s_id, data in all_results[policy_name].items()
                          if data["scenario_type"] == stype]
            if type_metrics:
                acc = np.mean([m["routing_accuracy"] for m in type_metrics])
                tok = np.mean([m["avg_tokens_per_turn"] for m in type_metrics])
                print(f"    {policy_name:<20} route_acc={acc:.4f} tokens/turn={tok:.0f}")

    # Adaptation speed for transition scenarios
    print("\n--- Adaptation Speed (transition scenarios) ---")
    for policy_name in policies:
        speeds = []
        for s_id, data in all_results[policy_name].items():
            if "transition" in data["scenario_type"]:
                adapt = data["metrics"].get("adaptation_speed")
                if adapt is not None:
                    speeds.append(adapt)
        if speeds:
            print(f"  {policy_name:<20} avg turns to adapt: {np.mean(speeds):.1f} "
                  f"(min={min(speeds)}, max={max(speeds)})")

    # Generate figures
    print("\nGenerating figures...")
    make_figures(all_results)

    # Save results
    # Strip turn_results for JSON serialization (too large)
    save_results = {}
    for policy_name, scenarios_data in all_results.items():
        save_results[policy_name] = {}
        for s_id, data in scenarios_data.items():
            save_results[policy_name][s_id] = {
                "scenario_type": data["scenario_type"],
                "transition_turn": data["transition_turn"],
                "metrics": data["metrics"],
                # Keep only route decisions from turn results
                "routes": [t["route"] for t in data["turn_results"]],
                "correct_routes": [t["correct_route"] for t in data["turn_results"]],
            }

    with open(OUT_DIR / "dialogue_agent_results.json", "w") as f:
        json.dump({"summary": summary, "per_scenario": save_results}, f, indent=2, default=str)

    print(f"\nResults saved to {OUT_DIR / 'dialogue_agent_results.json'}")


if __name__ == "__main__":
    main()
