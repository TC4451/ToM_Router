"""Multi-turn dialogue agent with reasoning policy selection."""

import time
from dataclasses import dataclass, field

from src.inference.policies import ReasoningPolicy
from src.models.experts import BaseExpert


@dataclass
class TurnRecord:
    """Record for a single dialogue turn."""
    turn_id: int
    user_message: str
    agent_response: str
    route: str  # "tom" or "social"
    requires_tom_gold: int  # ground truth if available, -1 if unknown
    tokens_in: int
    tokens_out: int
    latency_ms: float
    correct_route: bool


class DialogueAgent:
    """Multi-turn dialogue agent that selects reasoning strategy per turn.

    Maintains conversation history and routes each turn to the appropriate
    expert based on the configured reasoning policy.
    """

    def __init__(
        self,
        policy: ReasoningPolicy,
        tom_expert: BaseExpert,
        social_expert: BaseExpert,
        history_window: int = 5,
    ):
        self.policy = policy
        self.tom_expert = tom_expert
        self.social_expert = social_expert
        self.history_window = history_window
        self.history: list[TurnRecord] = []
        self.turn_count = 0

    def _build_context(self, current_message: str) -> tuple[str, str]:
        """Build context from conversation history + current message.

        Returns (context, question) matching the router's training format.
        """
        # Use recent history as context
        history_turns = self.history[-self.history_window:]
        context_parts = []
        for turn in history_turns:
            context_parts.append(f"User: {turn.user_message}")
            context_parts.append(f"Agent: {turn.agent_response}")

        context = " ".join(context_parts) if context_parts else ""
        question = current_message
        return context, question

    def respond(self, user_message: str, gold_label: int = -1) -> dict:
        """Process one turn of dialogue.

        Args:
            user_message: the user's input
            gold_label: ground truth requires_tom (0/1), -1 if unknown

        Returns:
            dict with response, route, metrics
        """
        context, question = self._build_context(user_message)

        # Decide route
        start = time.time()
        route = self.policy.decide(context, question)

        # Get expert response
        if route == "tom":
            expert_out = self.tom_expert.predict(context, question)
        else:
            expert_out = self.social_expert.predict(context, question)

        latency = (time.time() - start) * 1000

        # Extract token counts from expert metadata
        tokens_in = expert_out.get("metadata", {}).get("tokens_in", len(context.split()) + len(question.split()))
        tokens_out = expert_out.get("metadata", {}).get("tokens_out", len(expert_out["answer"].split()))

        # Record turn
        correct_route = (route == ("tom" if gold_label == 1 else "social")) if gold_label >= 0 else True
        self.turn_count += 1

        record = TurnRecord(
            turn_id=self.turn_count,
            user_message=user_message,
            agent_response=expert_out["answer"],
            route=route,
            requires_tom_gold=gold_label,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency,
            correct_route=correct_route,
        )
        self.history.append(record)

        return {
            "response": expert_out["answer"],
            "route": route,
            "correct_route": correct_route,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency,
            "turn_id": self.turn_count,
            "confidence": expert_out.get("confidence", None),
        }

    def reset(self):
        """Reset conversation state."""
        self.history = []
        self.turn_count = 0

    def get_metrics(self) -> dict:
        """Compute aggregate metrics for the conversation."""
        if not self.history:
            return {}

        total_tokens = sum(t.tokens_in + t.tokens_out for t in self.history)
        total_latency = sum(t.latency_ms for t in self.history)
        n_tom = sum(1 for t in self.history if t.route == "tom")
        n_social = sum(1 for t in self.history if t.route == "social")

        # Routing accuracy (only for turns with gold labels)
        labeled = [t for t in self.history if t.requires_tom_gold >= 0]
        route_acc = sum(t.correct_route for t in labeled) / len(labeled) if labeled else None

        return {
            "policy": self.policy.name,
            "n_turns": len(self.history),
            "n_tom_routes": n_tom,
            "n_social_routes": n_social,
            "total_tokens": total_tokens,
            "avg_tokens_per_turn": total_tokens / len(self.history),
            "total_latency_ms": total_latency,
            "avg_latency_ms": total_latency / len(self.history),
            "routing_accuracy": route_acc,
            "tom_ratio": n_tom / len(self.history),
        }
