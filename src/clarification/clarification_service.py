"""
Counter-question decision tree.
Orchestrates ambiguity detection and generates clarification questions.
"""
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from src.retrieval.rrf_fusion import HybridResult
from src.clarification.classifier import detect_ambiguity, AmbiguityResult
from src.clarification.templates import (
    build_counter_question,
    build_options,
)
from src.knowledge_generator.domains import DOMAIN_SHORT_LABELS
from config.settings import settings


class ClarifyAction(str, Enum):
    ANSWER = "answer"
    COUNTER_QUESTION = "counter_question"


@dataclass
class ClarifyDecision:
    """Final decision from the clarification system."""
    action: ClarifyAction
    domain: str | None = None  # Best domain if answering directly
    counter_question: str | None = None  # Question to ask user
    options: list[dict] = field(default_factory=list)  # [{label, domain, refined_query}]
    detected_domains: list[str] = field(default_factory=list)
    reason: str = ""


class ClarificationEngine:
    """
    Full clarification pipeline.
    Takes hybrid search results → decides whether to answer or ask.
    """

    def decide(
        self,
        query: str,
        results: list[HybridResult],
    ) -> ClarifyDecision:
        """
        Main decision entry point.
        Returns a ClarifyDecision with either answer domain or counter-question.
        """
        # Run ambiguity detection
        ambiguity = detect_ambiguity(query, results)

        if not ambiguity.is_ambiguous:
            return ClarifyDecision(
                action=ClarifyAction.ANSWER,
                domain=ambiguity.dominant_domain,
                detected_domains=ambiguity.detected_domains,
                reason=ambiguity.reason,
            )

        # Build counter-question
        domains = ambiguity.detected_domains
        if not domains:
            domains = list(set(r.domain for r in results[:5]))

        counter_q = build_counter_question(query, domains)
        options = build_options(query, domains)

        return ClarifyDecision(
            action=ClarifyAction.COUNTER_QUESTION,
            counter_question=counter_q,
            options=options,
            detected_domains=domains,
            reason=ambiguity.reason,
        )

    def refine_query(self, original_query: str, selected_domain: str) -> str:
        """
        Refine a query after user selects a domain.
        Adds domain context to improve retrieval.
        """
        domain_labels = DOMAIN_SHORT_LABELS
        label = domain_labels.get(selected_domain, selected_domain)
        return f"[{label}] {original_query}"
