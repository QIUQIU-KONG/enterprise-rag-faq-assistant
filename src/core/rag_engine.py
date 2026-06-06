"""
Platform-agnostic RAG Engine — the single entry point for all queries.

Wires together retrieval, reranking, generation, and clarification.
Exposes a pure query() method with no HTTP or platform-specific logic.
All platform adapters call this engine, never the retrieval pipeline directly.
"""
from loguru import logger

from src.retrieval.bm25_retriever import BM25Index
from src.retrieval.dense_retriever import VectorStore
from src.retrieval.hybrid_searcher import HybridSearcher
from src.retrieval.reranker import Reranker, RankedResult
from src.generation.llm_client import create_llm_client, LLMClient, LLMResponse
from src.generation.prompt_builder import build_rag_prompt
from src.clarification.clarification_service import ClarificationEngine, ClarifyDecision
from config.settings import settings


class RAGEngine:
    """
    Central RAG engine that wires together all components.
    Provides a single query() entry point for the full pipeline:

        query → clarify → hybrid search → rerank → generate → answer

    This class is platform-agnostic: it takes a string query and returns
    a dict with answer/sources/action. All platform-specific formatting
    is handled by adapters, not here.
    """

    def __init__(self):
        self.bm25 = BM25Index()
        self.vector_store = VectorStore()
        self.hybrid_searcher = HybridSearcher(self.bm25, self.vector_store)
        self.reranker = Reranker()
        self.llm: LLMClient | None = None
        self.clarification = ClarificationEngine()
        self._initialized = False

    def initialize(self):
        """Load indexes and models. Call once on startup."""
        if self._initialized:
            return

        # Load BM25 index
        bm25_path = settings.PROCESSED_DATA_DIR / "bm25_index.pkl"
        if bm25_path.exists():
            self.bm25.load(bm25_path)
            logger.info("BM25 index loaded")
        else:
            logger.warning(f"BM25 index not found at {bm25_path}. Run ingestion first.")

        # Load vector store
        self.vector_store.load()
        logger.info(f"Vector store loaded: {self.vector_store.count()} chunks")

        # Initialize LLM
        self.llm = create_llm_client()
        logger.info(f"LLM client initialized: {settings.LLM_PROVIDER}")

        self._initialized = True
        logger.info("RAG engine initialized")

    def query(
        self,
        query: str,
        domain_filter: str | None = None,
    ) -> dict:
        """
        Full query pipeline: clarify → search → rerank → generate.

        Args:
            query: User's question text.
            domain_filter: Optional domain to restrict search (skips clarification).

        Returns:
            dict with keys: answer, sources, action, counter_question, options, domain, model.
        """
        if not self._initialized:
            self.initialize()

        decision = None

        # Step 1: Hybrid search for clarification (skip if domain already specified)
        if domain_filter:
            decision = None  # Skip clarification when domain is pre-selected
        else:
            clarify_results = self.hybrid_searcher.search_with_clarify(query)
            decision = self.clarification.decide(query, clarify_results)

            logger.info(
                f"Clarify decision: {decision.action} "
                f"(domains={decision.detected_domains}, reason={decision.reason})"
            )

            # Step 2: If counter-question needed, return early
            if decision.action == "counter_question":
                return {
                    "answer": "",
                    "sources": [],
                    "action": "counter_question",
                    "counter_question": decision.counter_question,
                    "options": decision.options,
                    "domain": None,
                    "model": "",
                }

        # Step 3: Full hybrid search
        search_results = self.hybrid_searcher.search(query, domain_filter=domain_filter)

        # Step 4: Rerank with cross-encoder (fallback to RRF scores)
        try:
            ranked = self.reranker.rerank(query, search_results)
            logger.debug(f"Reranked {len(search_results)} results → top {len(ranked)}")
        except Exception as e:
            logger.warning(f"Reranker unavailable ({e}), using RRF scores directly")
            ranked = [
                RankedResult(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    domain=r.domain,
                    sub_topic=r.sub_topic,
                    title=r.title,
                    source_file=r.source_file,
                    relevance_score=r.rrf_score,
                    rrf_score=r.rrf_score,
                )
                for r in search_results[: settings.FINAL_TOP_K]
            ]

        # Handle empty search results BEFORE calling LLM
        if not ranked:
            return {
                "answer": "目前知识库中暂无相关信息，建议联系相关部门确认。",
                "sources": [],
                "action": "answer",
                "counter_question": None,
                "options": [],
                "domain": decision.domain if decision else domain_filter,
                "model": "none",
            }

        # Step 5: Generate answer (with fallback on LLM failure)
        try:
            answer = self._generate(query, ranked)
            model = answer.model
            answer_text = answer.content
            # Validate answer quality
            if not self._validate_answer(answer_text, ranked):
                logger.warning("Answer quality check failed — possible hallucination")
                answer_text += "\n\n⚠️ 该回答可能未完全基于知识库内容，请参考下方来源文档。"
        except Exception as e:
            logger.error(f"LLM generation failed: {type(e).__name__}")
            # Fallback: return retrieved document summaries
            model = "fallback"
            answer_text = self._build_fallback_answer(query, ranked)

        # Step 6: Build response
        sources = [
            {
                "title": r.title,
                "domain": r.domain,
                "content": r.content,
                "source_file": r.source_file,
                "score": round(r.relevance_score, 4),
            }
            for r in ranked
        ]

        return {
            "answer": answer_text,
            "sources": sources,
            "action": "answer",
            "counter_question": None,
            "options": [],
            "domain": decision.domain if decision else domain_filter,
            "model": model,
        }

    def _generate(
        self,
        query: str,
        contexts: list[RankedResult],
    ) -> LLMResponse:
        """Generate answer from ranked contexts."""
        ctx_list = [
            {
                "title": c.title,
                "domain": c.domain,
                "content": c.content,
            }
            for c in contexts
        ]

        system_prompt, user_prompt = build_rag_prompt(query, ctx_list)
        return self.llm.generate(system_prompt, user_prompt)

    def _validate_answer(self, answer: str, contexts: list[RankedResult]) -> bool:
        """Quick check: does the answer reference at least some context content?"""
        if len(answer) < 20:
            return False
        combined_context = " ".join(c.content for c in contexts)
        # Use 2-char bigrams to check content overlap (same approach as BM25 tokenizer)
        import re
        answer_words = set(re.findall(r"[一-鿿]{2}", answer))
        context_words = set(re.findall(r"[一-鿿]{2}", combined_context))
        if not answer_words:
            return True  # Can't validate non-Chinese answers
        overlap = answer_words & context_words
        return len(overlap) >= max(2, len(answer_words) * 0.3)

    def _build_fallback_answer(self, query: str, contexts: list[RankedResult]) -> str:
        """Build a fallback answer from retrieved documents when LLM is unavailable."""
        lines = [
            "⚠️ AI 服务暂时不可用，以下是根据知识库检索到的相关内容摘要：",
            "",
        ]
        for i, ctx in enumerate(contexts[:3], 1):
            # Show first 150 chars of each context as a preview
            preview = ctx.content[:150].replace("\n", " ").strip()
            lines.append(f"{i}. 📄《{ctx.title}》")
            lines.append(f"   {preview}...")
            lines.append("")

        lines.append("---")
        lines.append("请稍后重试，或直接查阅上述来源文档获取完整信息。")
        return "\n".join(lines)

    def index_size(self) -> int:
        """Return total number of indexed chunks."""
        return self.vector_store.count()


# Global engine singleton
_engine: RAGEngine | None = None


def get_rag_engine() -> RAGEngine:
    """Get or create the global RAG engine singleton."""
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine
