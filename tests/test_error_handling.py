"""
Tests for error handling and graceful degradation.
Verifies the system doesn't crash under adverse conditions.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLLMErrorHandling:
    """Verify LLM failures are handled gracefully."""

    def test_mock_returns_without_crashing(self):
        """Debug/mock LLM should always return something."""
        from src.generation.llm_client import create_llm_client, MockClient
        from config.settings import settings

        # Force debug mode for this test
        old_provider = settings.LLM_PROVIDER
        try:
            settings.LLM_PROVIDER = "debug"
            client = create_llm_client()
            assert isinstance(client, MockClient)

            response = client.generate("system", "user query")
            assert response.content, "Should return non-empty content"
            assert response.model == "mock"
        finally:
            settings.LLM_PROVIDER = old_provider

    def test_unknown_provider_raises(self):
        """Unknown provider should raise clear error."""
        from src.generation.llm_client import create_llm_client
        from config.settings import settings

        old_provider = settings.LLM_PROVIDER
        try:
            settings.LLM_PROVIDER = "nonexistent_provider"
            with pytest.raises(ValueError, match="Unknown LLM provider"):
                create_llm_client()
        finally:
            settings.LLM_PROVIDER = old_provider


class TestEmptySearchHandling:
    """Verify empty search results are handled without LLM calls."""

    def test_empty_results_no_answer(self):
        """No search results → fallback message without hitting LLM."""
        from src.core.rag_engine import RAGEngine
        from src.retrieval.reranker import RankedResult

        engine = RAGEngine.__new__(RAGEngine)
        engine.llm = None  # No LLM
        engine._initialized = True

        result = engine._build_fallback_answer("测试问题", [])
        assert "AI 服务暂时不可用" in result or "暂无" in result

    def test_fallback_answer_uses_contexts(self):
        """Fallback answer should include context previews."""
        from src.retrieval.reranker import RankedResult

        contexts = [
            RankedResult(
                chunk_id="1", content="办理签证需要护照原件及复印件",
                domain="malaysia_visa", sub_topic="", title="签证材料",
                source_file="visa.md", relevance_score=0.9, rrf_score=0.03,
            ),
            RankedResult(
                chunk_id="2", content="马来西亚使用英标三孔插座",
                domain="travel_tips", sub_topic="", title="物品准备",
                source_file="packing.md", relevance_score=0.5, rrf_score=0.02,
            ),
        ]

        from src.core.rag_engine import RAGEngine
        engine = RAGEngine.__new__(RAGEngine)
        answer = engine._build_fallback_answer("插座", contexts)

        assert "签证材料" in answer, "Should include first context title"
        assert "物品准备" in answer, "Should include second context title"
        assert "AI 服务暂时不可用" in answer


class TestAnswerValidation:
    """Verify the answer quality validator."""

    def test_good_answer_validates(self):
        """Answer with context keywords → valid."""
        from src.core.rag_engine import RAGEngine
        from src.retrieval.reranker import RankedResult

        contexts = [
            RankedResult(
                chunk_id="1",
                content="办理签证需要护照原件和两寸照片",
                domain="malaysia_visa", sub_topic="", title="签证材料",
                source_file="visa.md", relevance_score=0.9, rrf_score=0.03,
            ),
        ]

        engine = RAGEngine.__new__(RAGEngine)
        # Answer uses keywords from context
        result = engine._validate_answer(
            "办理签证需要准备护照原件和两寸照片，请提前准备",
            contexts,
        )
        assert result is True

    def test_hallucinated_answer_fails(self):
        """Answer with no context keywords → flagged."""
        from src.core.rag_engine import RAGEngine
        from src.retrieval.reranker import RankedResult

        contexts = [
            RankedResult(
                chunk_id="1",
                content="办理签证需要护照原件和两寸照片",
                domain="malaysia_visa", sub_topic="", title="签证材料",
                source_file="visa.md", relevance_score=0.9, rrf_score=0.03,
            ),
        ]

        engine = RAGEngine.__new__(RAGEngine)
        # Answer talks about something completely different
        result = engine._validate_answer(
            "您需要先去银行开户然后转账手续费是五百元整",
            contexts,
        )
        assert result is False, "Answer unrelated to context should fail validation"

    def test_too_short_answer_fails(self):
        """Very short answer → invalid."""
        from src.core.rag_engine import RAGEngine
        from src.retrieval.reranker import RankedResult

        engine = RAGEngine.__new__(RAGEngine)
        result = engine._validate_answer("好的", [])
        assert result is False, "Very short answer should be flagged"


class TestSafeChatResponse:
    """Verify the API-level error wrapper."""

    def test_safe_wrapper_catches_exceptions(self):
        """Wrapper should return error dict instead of raising."""
        from src.api.routes import _safe_chat_response

        class BrokenEngine:
            def query(self, **kwargs):
                raise RuntimeError("Simulated catastrophic failure")

        result = _safe_chat_response(BrokenEngine(), "test query")
        assert result["action"] == "answer"
        assert "错误" in result["answer"], "Should return Chinese error message"
        assert result["model"] == "error"
