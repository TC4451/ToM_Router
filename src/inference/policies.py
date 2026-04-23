"""Reasoning policies for the dialogue agent."""

from abc import ABC, abstractmethod


class ReasoningPolicy(ABC):
    """Base class for reasoning policies."""

    @abstractmethod
    def decide(self, context: str, question: str) -> str:
        """Decide which expert to use: 'tom' or 'social'."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class AlwaysToMPolicy(ReasoningPolicy):
    """Always route to the ToM expert. Maximum quality, maximum cost."""

    def decide(self, context: str, question: str) -> str:
        return "tom"

    @property
    def name(self) -> str:
        return "always_tom"


class GeneralSocialPolicy(ReasoningPolicy):
    """Always route to the social expert. Minimum cost, misses ToM cases."""

    def decide(self, context: str, question: str) -> str:
        return "social"

    @property
    def name(self) -> str:
        return "general_social"


class AdaptiveRouterPolicy(ReasoningPolicy):
    """Use the trained router to decide per-turn. Best cost-quality tradeoff."""

    def __init__(self, router_pipeline):
        self.router = router_pipeline

    def decide(self, context: str, question: str) -> str:
        result = self.router.predict(context, question)
        return result["route"]

    @property
    def name(self) -> str:
        return "adaptive_router"
