"""
Domain classifier and ambiguity detector for the counter-question system.
Analyzes search results to determine if a query is single-domain or ambiguous.
"""
from dataclasses import dataclass, field
from collections import Counter
from loguru import logger

from src.retrieval.rrf_fusion import HybridResult
from config.settings import settings


@dataclass
class DomainDistribution:
    """Domain distribution in search results."""
    counts: dict[str, int]
    ratios: dict[str, float]
    dominant_domain: str
    dominant_ratio: float
    domain_count: int  # Number of distinct domains found


def analyze_domain_distribution(results: list[HybridResult]) -> DomainDistribution:
    """
    Analyze which domains appear in the search results.
    Returns distribution stats for decision-making.
    """
    if not results:
        return DomainDistribution(
            counts={},
            ratios={},
            dominant_domain="",
            dominant_ratio=0.0,
            domain_count=0,
        )

    domains = [r.domain for r in results]
    counts = dict(Counter(domains))
    total = len(results)
    ratios = {d: c / total for d, c in counts.items()}

    dominant = max(counts, key=counts.get)
    return DomainDistribution(
        counts=counts,
        ratios=ratios,
        dominant_domain=dominant,
        dominant_ratio=ratios[dominant],
        domain_count=len(counts),
    )


@dataclass
class AmbiguityResult:
    """Result of ambiguity analysis."""
    is_ambiguous: bool
    action: str  # "answer" | "counter_question"
    dominant_domain: str | None
    detected_domains: list[str] = field(default_factory=list)
    reason: str = ""


def detect_ambiguity(
    query: str,
    results: list[HybridResult],
) -> AmbiguityResult:
    """
    Detect whether a query is ambiguous and needs clarification.

    Decision logic:
    1. If ≥80% results from one domain AND top score gap ≥ 0.12 → ANSWER
    2. If ≥80% results from one domain BUT score gap < 0.12 → LLM check
    3. If <80% from one domain → COUNTER-QUESTION (multi-domain)
    """
    if not results:
        return AmbiguityResult(
            is_ambiguous=True,
            action="counter_question",
            dominant_domain=None,
            reason="No search results found",
        )

    dist = analyze_domain_distribution(results)

    # Case 1: Strong single domain signal
    if dist.dominant_ratio >= settings.DOMAIN_RATIO_THRESHOLD:
        # Check score gap between top 2 results
        gap = _compute_score_gap(results, dist.dominant_domain)

        if gap >= settings.SCORE_GAP_THRESHOLD:
            return AmbiguityResult(
                is_ambiguous=False,
                action="answer",
                dominant_domain=dist.dominant_domain,
                detected_domains=list(dist.counts.keys()),
                reason=f"Clear single domain ({dist.dominant_domain}), gap={gap:.3f}",
            )
        else:
            # Scores too close — borderline case.
            # TODO: Integrate LLM-based judgment here for better accuracy.
            return AmbiguityResult(
                is_ambiguous=True,
                action="counter_question",
                dominant_domain=dist.dominant_domain,
                detected_domains=list(dist.counts.keys()),
                reason=f"Score gap too small ({gap:.3f}), needs clarification",
            )

    # Case 2: Multi-domain
    # Only include domains with meaningful presence (>20% of results, minimum 3)
    total = sum(dist.counts.values())
    min_count = max(3, int(total * 0.2))
    detected = [d for d, c in dist.counts.items() if c >= min_count]
    return AmbiguityResult(
        is_ambiguous=True,
        action="counter_question",
        dominant_domain=dist.dominant_domain,
        detected_domains=detected,
        reason=f"Multi-domain: {dist.ratios}",
    )


def _compute_score_gap(results: list[HybridResult], dominant_domain: str) -> float:
    """Compute score gap between best dominant-domain result and best non-dominant."""
    best_dom = 0.0
    best_other = 0.0

    for r in results:
        if r.domain == dominant_domain:
            best_dom = max(best_dom, r.rrf_score)
        else:
            best_other = max(best_other, r.rrf_score)

    if best_dom == 0:
        return 0.0
    return best_dom - best_other
