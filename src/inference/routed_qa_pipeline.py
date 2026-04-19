"""Routed QA pipeline — routes queries to ToM or Social expert via the router."""

from src.inference.router_pipeline import RouterPipeline
from src.models.experts import BaseExpert


class RoutedQAPipeline:
    """End-to-end routed QA system.

    Routes queries to either a ToM expert or Social expert based on
    the student router's prediction.
    """

    def __init__(
        self,
        router: RouterPipeline,
        tom_expert: BaseExpert,
        social_expert: BaseExpert,
        uncertainty_margin: float = 0.1,
        fallback_mode: str = "default_tom",
    ):
        self.router = router
        self.tom_expert = tom_expert
        self.social_expert = social_expert
        self.uncertainty_margin = uncertainty_margin
        self.fallback_mode = fallback_mode

    def predict(self, context: str, question: str) -> dict:
        """Route and answer a query."""
        router_out = self.router.predict(context, question)
        prob_tom = router_out["prob_tom"]

        # Check uncertainty
        is_uncertain = router_out["uncertainty"] > (1 - 2 * self.uncertainty_margin)

        if is_uncertain and self.fallback_mode == "both":
            # Query both experts
            tom_out = self.tom_expert.predict(context, question)
            social_out = self.social_expert.predict(context, question)
            # Pick higher confidence
            if tom_out.get("confidence", 0) >= social_out.get("confidence", 0):
                expert_out = tom_out
                route = "tom_fallback"
            else:
                expert_out = social_out
                route = "social_fallback"
        elif is_uncertain and self.fallback_mode == "default_tom":
            expert_out = self.tom_expert.predict(context, question)
            route = "tom_fallback"
        elif prob_tom >= self.router.threshold:
            expert_out = self.tom_expert.predict(context, question)
            route = "tom"
        else:
            expert_out = self.social_expert.predict(context, question)
            route = "social"

        return {
            "route": route,
            "prob_tom": prob_tom,
            "uncertainty": router_out["uncertainty"],
            "answer": expert_out["answer"],
            "expert_confidence": expert_out.get("confidence", None),
            "expert_metadata": expert_out.get("metadata", {}),
        }
