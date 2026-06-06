"""
Reciprocal Rank Fusion (RRF) algorithm.

Combines ranked results from BM25 and dense retrieval by scoring
documents based on their reciprocal rank in each result list.
Documents found by both methods receive boosted scores.

RRF_score(d) = Sum(1 / (k + rank_i(d)))
"""
from dataclasses import dataclass, field
from loguru import logger

from src.retrieval.bm25_retriever import BM25Result


@dataclass
class HybridResult:
    """Unified search result from hybrid retrieval."""
    chunk_id: str
    content: str
    domain: str
    sub_topic: str
    title: str
    source_file: str
    bm25_score: float
    dense_score: float
    rrf_score: float
    metadata: dict = field(default_factory=dict)


def rrf_fuse(
    bm25_results: list[BM25Result],
    dense_results: list[dict],
    rrf_k: int = 60,
) -> list[HybridResult]:
    """
    Fuse BM25 and dense retrieval results using Reciprocal Rank Fusion.

    Args:
        bm25_results: Ranked results from BM25 search.
        dense_results: Ranked results from dense vector search.
        rrf_k: RRF constant (default 60, standard value from literature).

    Returns:
        List of HybridResult sorted by RRF score descending.
    """
    # Build unified result map
    result_map: dict[str, HybridResult] = {}

    # Process BM25 results
    for rank, r in enumerate(bm25_results, start=1):
        result_map[r.chunk_id] = HybridResult(
            chunk_id=r.chunk_id,
            content=r.content,
            domain=r.domain,
            sub_topic=r.sub_topic,
            title=r.title,
            source_file=r.source_file,
            bm25_score=r.score,
            dense_score=0.0,
            rrf_score=0.0,
        )

    # Process dense results
    for rank, r in enumerate(dense_results, start=1):
        chunk_id = r["id"]
        if chunk_id in result_map:
            result_map[chunk_id].dense_score = r["score"]
            result_map[chunk_id].metadata = r.get("metadata", {})
        else:
            metadata = r.get("metadata", {})
            result_map[chunk_id] = HybridResult(
                chunk_id=chunk_id,
                content=r.get("content", ""),
                domain=metadata.get("domain", ""),
                sub_topic=metadata.get("sub_topic", ""),
                title=metadata.get("title", ""),
                source_file=metadata.get("source_file", ""),
                bm25_score=0.0,
                dense_score=r["score"],
                rrf_score=0.0,
                metadata=metadata,
            )

    # Compute RRF scores
    for chunk_id, result in result_map.items():
        rrf = 0.0
        # BM25 contribution
        for rank, r in enumerate(bm25_results, start=1):
            if r.chunk_id == chunk_id:
                rrf += 1.0 / (rrf_k + rank)
                break
        # Dense contribution
        for rank, r in enumerate(dense_results, start=1):
            if r["id"] == chunk_id:
                rrf += 1.0 / (rrf_k + rank)
                break
        result.rrf_score = rrf

    # Sort by RRF score descending
    fused = sorted(result_map.values(), key=lambda x: x.rrf_score, reverse=True)
    return fused
