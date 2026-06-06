"""
Tests for the clarification engine: domain distribution analysis,
ambiguity detection, and counter-question generation.

What we verify:
  1. Single-domain results → direct answer
  2. Multi-domain results → counter-question
  3. Empty results handled gracefully
  4. Domain distribution is calculated correctly
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Test helpers
# ============================================================

def _make_result(chunk_id: str, domain: str, rrf_score: float = 0.03):
    """Create a HybridResult for testing."""
    from src.retrieval.rrf_fusion import HybridResult
    return HybridResult(
        chunk_id=chunk_id,
        content=f"Content of {chunk_id} in {domain}",
        domain=domain,
        sub_topic="test",
        title=f"Doc {chunk_id}",
        source_file=f"{domain}/doc.md",
        bm25_score=rrf_score,
        dense_score=rrf_score,
        rrf_score=rrf_score,
    )


# ============================================================
# Domain Distribution Tests
# ============================================================

class TestDomainDistribution:
    """Verify domain counting and ratio calculation."""

    def test_pure_single_domain(self):
        """All results from same domain → 100% dominant ratio."""
        from src.clarification.classifier import analyze_domain_distribution

        results = [
            _make_result("a", "travel_tips", 0.05),
            _make_result("b", "travel_tips", 0.04),
            _make_result("c", "travel_tips", 0.03),
        ]
        dist = analyze_domain_distribution(results)

        assert dist.dominant_domain == "travel_tips"
        assert dist.dominant_ratio == 1.0
        assert dist.domain_count == 1

    def test_mixed_two_domains(self):
        """50/50 split → two domains, no clear dominant."""
        from src.clarification.classifier import analyze_domain_distribution

        results = [
            _make_result("a", "travel_tips", 0.05),
            _make_result("b", "malaysia_visa", 0.05),
            _make_result("c", "travel_tips", 0.04),
            _make_result("d", "malaysia_visa", 0.04),
        ]
        dist = analyze_domain_distribution(results)

        assert dist.domain_count == 2
        assert dist.dominant_ratio == 0.5

    def test_three_domains(self):
        """Three domains → multi-domain detection."""
        from src.clarification.classifier import analyze_domain_distribution

        results = [
            _make_result("a", "travel_tips", 0.05),
            _make_result("b", "malaysia_visa", 0.04),
            _make_result("c", "project_applications", 0.03),
        ]
        dist = analyze_domain_distribution(results)

        assert dist.domain_count == 3
        assert dist.dominant_ratio == 1.0 / 3.0

    def test_empty_results(self):
        """No results → zero domain count."""
        from src.clarification.classifier import analyze_domain_distribution

        dist = analyze_domain_distribution([])

        assert dist.domain_count == 0
        assert dist.dominant_domain == ""


# ============================================================
# Ambiguity Detection Tests
# ============================================================

class TestAmbiguityDetection:
    """Verify the ambiguity detection decision logic."""

    def test_clear_single_domain_not_ambiguous(self):
        """≥80% from one domain + large score gap → NOT ambiguous."""
        from src.clarification.classifier import detect_ambiguity

        results = [
            _make_result("a", "malaysia_visa", 0.20),  # dominant, high score
            _make_result("b", "malaysia_visa", 0.18),
            _make_result("c", "malaysia_visa", 0.16),
            _make_result("d", "malaysia_visa", 0.14),
            _make_result("e", "travel_tips", 0.02),    # non-dominant, low score
            # Score gap = 0.20 - 0.02 = 0.18 > 0.12 threshold
        ]
        result = detect_ambiguity("签证材料", results)

        assert result.is_ambiguous is False
        assert result.action == "answer"
        assert result.dominant_domain == "malaysia_visa"

    def test_small_gap_triggers_clarification(self):
        """Same domain ratio but small score gap → still ambiguous."""
        from src.clarification.classifier import detect_ambiguity

        results = [
            _make_result("a", "malaysia_visa", 0.08),
            _make_result("b", "malaysia_visa", 0.07),
            _make_result("c", "malaysia_visa", 0.06),
            _make_result("d", "malaysia_visa", 0.05),
            _make_result("e", "travel_tips", 0.01),
            # Gap = 0.07 < 0.12 threshold → needs clarification
        ]
        result = detect_ambiguity("签证材料", results)

        # Even though 80% is from one domain, small gap → counter_question
        assert result.is_ambiguous is True
        assert result.action == "counter_question"

    def test_multi_domain_is_ambiguous(self):
        """Less than 80% from any domain → ambiguous."""
        from src.clarification.classifier import detect_ambiguity

        results = [
            _make_result("a", "travel_tips", 0.05),
            _make_result("b", "malaysia_visa", 0.05),
            _make_result("c", "travel_tips", 0.04),
            _make_result("d", "malaysia_visa", 0.04),
            _make_result("e", "travel_tips", 0.03),
            _make_result("f", "malaysia_visa", 0.03),
        ]
        result = detect_ambiguity("准备什么", results)

        assert result.is_ambiguous is True
        assert result.action == "counter_question"
        assert len(result.detected_domains) >= 2

    def test_empty_results_ambiguous(self):
        """No results at all → ambiguous (no info to answer)."""
        from src.clarification.classifier import detect_ambiguity

        result = detect_ambiguity("some query", [])

        assert result.is_ambiguous is True


# ============================================================
# Clarification Engine Tests
# ============================================================

class TestClarificationEngine:
    """Verify the full clarification decision pipeline."""

    def test_decide_answer_when_clear(self):
        """Clear single domain + large gap → decision is ANSWER."""
        from src.clarification.clarification_service import ClarificationEngine, ClarifyAction

        results = [
            _make_result("a", "malaysia_visa", 0.25),
            _make_result("b", "malaysia_visa", 0.22),
            _make_result("c", "malaysia_visa", 0.20),
            _make_result("d", "malaysia_visa", 0.18),
            _make_result("e", "travel_tips", 0.02),
            # Gap = 0.23 >> 0.12 threshold
        ]
        engine = ClarificationEngine()
        decision = engine.decide("签证需要什么材料", results)

        assert decision.action == ClarifyAction.ANSWER
        assert decision.domain == "malaysia_visa"

    def test_decide_counter_question_when_ambiguous(self):
        """Multi-domain → decision is COUNTER_QUESTION."""
        from src.clarification.clarification_service import ClarificationEngine, ClarifyAction

        results = [
            _make_result("a", "travel_tips", 0.05),
            _make_result("b", "malaysia_visa", 0.05),
        ]
        engine = ClarificationEngine()
        decision = engine.decide("需要什么材料", results)

        assert decision.action == ClarifyAction.COUNTER_QUESTION
        assert decision.counter_question is not None
        assert len(decision.options) >= 2

    def test_refine_query_adds_domain_context(self):
        """Refine query should prepend domain label."""
        from src.clarification.clarification_service import ClarificationEngine

        engine = ClarificationEngine()
        refined = engine.refine_query("需要什么材料", "malaysia_visa")

        assert "马来西亚" in refined
        assert "需要什么材料" in refined
