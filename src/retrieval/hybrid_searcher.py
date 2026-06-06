"""
Hybrid search combining BM25 (keyword) and Dense (semantic) retrieval.

Orchestrates parallel retrieval from both indexes and delegates
result fusion to the extracted rrf_fusion module.
"""
from loguru import logger

from src.retrieval.bm25_retriever import BM25Index
from src.retrieval.dense_retriever import VectorStore
from src.retrieval.rrf_fusion import HybridResult, rrf_fuse
from config.settings import settings


class HybridSearcher:
    """
    Combines BM25 and dense retrieval via Reciprocal Rank Fusion.
    Documents found by both methods get boosted scores.
    """

    def __init__(self, bm25: BM25Index, vector_store: VectorStore):
        self.bm25 = bm25
        self.vector_store = vector_store
        self.rrf_k = settings.RRF_K

    def search(
        self,
        query: str,
        bm25_top_k: int | None = None,
        dense_top_k: int | None = None,
        domain_filter: str | None = None,
    ) -> list[HybridResult]:
        """
        Execute hybrid search and fuse results with RRF.
        Returns ranked list of HybridResult.
        """
        bm25_top_k = bm25_top_k or settings.BM25_TOP_K
        dense_top_k = dense_top_k or settings.DENSE_TOP_K

        # Parallel retrieval (sequential for now, can be parallelized later)
        bm25_results = self.bm25.search(query, top_k=bm25_top_k)
        dense_results = self.vector_store.search(
            query, top_k=dense_top_k, domain_filter=domain_filter
        )

        # Fuse with RRF (delegated to extracted module)
        fused = rrf_fuse(bm25_results, dense_results, self.rrf_k)

        logger.debug(
            f"Hybrid search: BM25={len(bm25_results)} + Dense={len(dense_results)} "
            f"→ Fused={len(fused)}"
        )
        return fused

    def search_with_clarify(self, query: str) -> list[HybridResult]:
        """
        Search with extra results for clarification analysis.
        Returns top_k=15 results for domain distribution analysis.
        """
        return self.search(
            query,
            bm25_top_k=settings.CLARIFY_TOP_K,
            dense_top_k=settings.CLARIFY_TOP_K,
        )
