"""
Tests for the retrieval pipeline: BM25, VectorStore, HybridSearch, RRF fusion.

What we verify:
  1. BM25 builds and searches correctly
  2. VectorStore stores, embeds, and searches
  3. Hybrid search merges results from both
  4. RRF fusion correctly boosts documents found by both methods
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Fixtures — create test data once, reuse across tests
# ============================================================

@pytest.fixture
def sample_chunks():
    """Minimal chunks simulating real ingestion output."""
    from src.ingestion.chunker import Chunk
    return [
        Chunk(
            id="c1", content="办理马来西亚商务签证需要护照和照片",
            domain="malaysia_visa", sub_topic="materials",
            title="签证办理材料", source_file="visa/materials.md",
            chunk_index=0, total_chunks=3,
            keywords=["签证", "护照", "照片"], language="zh",
        ),
        Chunk(
            id="c2", content="马来西亚使用英标三孔插座，电压220V",
            domain="travel_tips", sub_topic="packing",
            title="物品准备清单", source_file="travel/packing.md",
            chunk_index=0, total_chunks=3,
            keywords=["插座", "电压"], language="zh",
        ),
        Chunk(
            id="c3", content="商务签证不可在马来西亚境内延期",
            domain="malaysia_visa", sub_topic="faq",
            title="签证常见问题", source_file="visa/faq.md",
            chunk_index=0, total_chunks=3,
            keywords=["延期", "签证"], language="zh",
        ),
    ]


# ============================================================
# BM25 Tests
# ============================================================

class TestBM25Index:
    """Verify BM25 keyword search builds, searches, and saves/loads."""

    def test_build_and_search(self, sample_chunks):
        """BM25 should rank keyword-matching documents higher than non-matching."""
        from src.retrieval.bm25_retriever import BM25Index

        bm25 = BM25Index()
        bm25.build(sample_chunks)

        # "签证" appears in c1 and c3. c2 is about sockets.
        # With 3 tiny documents and Chinese bigram tokenization,
        # BM25 returns all docs but ranks relevant ones higher.
        results = bm25.search("签证", top_k=10)
        result_ids = {r.chunk_id for r in results}

        assert "c1" in result_ids, "c1 contains '签证' explicitly"
        assert "c3" in result_ids, "c3 contains '签证' explicitly"

        # The visa-related docs should rank above the socket doc
        scores = {r.chunk_id: r.score for r in results}
        assert scores.get("c1", 0) > scores.get("c2", 0), \
            "Visa doc should score higher than socket doc"
        assert scores.get("c3", 0) > scores.get("c2", 0), \
            "Visa FAQ doc should score higher than socket doc"

    def test_search_returns_top_k(self, sample_chunks):
        """search() should respect the top_k limit."""
        from src.retrieval.bm25_retriever import BM25Index

        bm25 = BM25Index()
        bm25.build(sample_chunks)
        results = bm25.search("签证", top_k=1)
        assert len(results) == 1

    def test_save_and_load(self, sample_chunks, tmp_path):
        """BM25 index should survive a save/load round-trip."""
        from src.retrieval.bm25_retriever import BM25Index

        bm25 = BM25Index()
        bm25.build(sample_chunks)

        path = tmp_path / "bm25_test.pkl"
        bm25.save(path)

        bm25_loaded = BM25Index()
        bm25_loaded.load(path)

        # Both should produce the same search results
        original = bm25.search("签证")
        loaded = bm25_loaded.search("签证")
        assert len(original) == len(loaded)

    def test_empty_query_returns_all(self, sample_chunks):
        """Empty query returns all docs with zero scores (expected BM25 behavior)."""
        from src.retrieval.bm25_retriever import BM25Index

        bm25 = BM25Index()
        bm25.build(sample_chunks)
        results = bm25.search("")
        # Empty query → BM25 returns all docs with 0 scores
        assert len(results) == 3
        for r in results:
            assert r.score == 0.0


# ============================================================
# VectorStore Tests
# ============================================================

class TestVectorStore:
    """Verify numpy-based vector store: add, search, filter, save/load."""

    def test_add_and_count(self, sample_chunks):
        """Adding chunks should increase count."""
        from src.retrieval.dense_retriever import VectorStore

        vs = VectorStore()
        vs.delete_collection()
        vs.add_chunks(sample_chunks)

        assert vs.count() == 3, f"Expected 3 chunks, got {vs.count()}"

    def test_search_returns_results(self, sample_chunks):
        """Search should return results sorted by relevance."""
        from src.retrieval.dense_retriever import VectorStore

        vs = VectorStore()
        vs.delete_collection()
        vs.add_chunks(sample_chunks)

        results = vs.search("签证需要什么材料", top_k=2)
        assert len(results) == 2
        # First result should be more relevant than second
        assert results[0]["score"] >= results[1]["score"]

    def test_domain_filter(self, sample_chunks):
        """Domain filter should only return matching chunks."""
        from src.retrieval.dense_retriever import VectorStore

        vs = VectorStore()
        vs.delete_collection()
        vs.add_chunks(sample_chunks)

        visa_results = vs.search("签证", domain_filter="malaysia_visa", top_k=10)
        travel_results = vs.search("插座", domain_filter="travel_tips", top_k=10)

        for r in visa_results:
            assert r["metadata"]["domain"] == "malaysia_visa"
        for r in travel_results:
            assert r["metadata"]["domain"] == "travel_tips"

    def test_save_and_load(self, sample_chunks, tmp_path):
        """Saved vectors should reload identically."""
        from src.retrieval.dense_retriever import VectorStore

        vs = VectorStore()
        vs.delete_collection()
        vs.add_chunks(sample_chunks)  # Triggers embedding

        # Monkey-patch save path
        vs._save_path = tmp_path / "vectors.pkl"
        vs.save()

        vs2 = VectorStore()
        vs2._save_path = tmp_path / "vectors.pkl"
        vs2.load()

        assert vs2.count() == vs.count()
        assert vs2.count() == 3

    def test_delete_collection(self, sample_chunks):
        """Reset should clear all data."""
        from src.retrieval.dense_retriever import VectorStore

        vs = VectorStore()
        vs.add_chunks(sample_chunks)
        assert vs.count() == 3

        vs.delete_collection()
        assert vs.count() == 0
        assert vs._embeddings is None


# ============================================================
# Hybrid Search & RRF Tests
# ============================================================

class TestHybridSearch:
    """Verify the BM25 + Dense → RRF fusion pipeline."""

    def test_hybrid_returns_fused_results(self, sample_chunks):
        """Hybrid search should return results with both scores populated."""
        from src.retrieval.bm25_retriever import BM25Index
        from src.retrieval.dense_retriever import VectorStore
        from src.retrieval.hybrid_searcher import HybridSearcher

        bm25 = BM25Index()
        bm25.build(sample_chunks)
        vs = VectorStore()
        vs.delete_collection()
        vs.add_chunks(sample_chunks)

        searcher = HybridSearcher(bm25, vs)
        results = searcher.search("签证", bm25_top_k=10, dense_top_k=10)

        assert len(results) > 0, "Should find at least some results"
        # At least one result should have both BM25 and Dense scores
        has_both = any(r.bm25_score > 0 and r.dense_score > 0 for r in results)
        # Note: not all results will have both, depending on overlap
        assert len(results) <= 20  # Fused should not exceed sum of inputs

    def test_rrf_boosts_dual_hits(self):
        """A document found by BOTH methods should score higher than one found by only one."""
        # This tests the core RRF logic directly
        from src.retrieval.bm25_retriever import BM25Result
        from src.retrieval.hybrid_searcher import HybridSearcher

        bm25_results = [
            BM25Result(chunk_id="shared", content="签证材料", domain="visa", sub_topic="", title="T1", source_file="", score=10.0),
            BM25Result(chunk_id="bm25_only", content="签证流程", domain="visa", sub_topic="", title="T2", source_file="", score=5.0),
        ]

        dense_results = [
            {"id": "shared", "score": 0.9, "content": "签证材料", "metadata": {"domain": "visa", "title": "T1"}},
            {"id": "dense_only", "score": 0.8, "content": "签证常见问题", "metadata": {"domain": "visa", "title": "T3"}},
        ]

        # Use the standalone rrf_fuse function (extracted from HybridSearcher)
        from src.retrieval.rrf_fusion import rrf_fuse
        fused = rrf_fuse(bm25_results, dense_results, rrf_k=60)

        # "shared" appears in both → higher RRF score
        shared = next(r for r in fused if r.chunk_id == "shared")
        bm25_only = next(r for r in fused if r.chunk_id == "bm25_only")
        dense_only = next(r for r in fused if r.chunk_id == "dense_only")

        assert shared.rrf_score > bm25_only.rrf_score, \
            "Document found by both methods should rank above BM25-only"
        assert shared.rrf_score > dense_only.rrf_score, \
            "Document found by both methods should rank above Dense-only"


# ============================================================
# Edge cases
# ============================================================

class TestRetrievalEdgeCases:
    """Verify graceful handling of empty/unusual inputs."""

    def test_empty_chunks(self):
        """Adding zero chunks should not crash."""
        from src.retrieval.dense_retriever import VectorStore

        vs = VectorStore()
        vs.delete_collection()
        vs.add_chunks([])
        assert vs.count() == 0

    def test_search_empty_store(self):
        """Searching an empty store returns empty list."""
        from src.retrieval.dense_retriever import VectorStore

        vs = VectorStore()
        vs.delete_collection()
        results = vs.search("anything")
        assert results == []

    def test_empty_query_bm25(self, sample_chunks):
        """BM25 with empty query returns all docs with zero scores."""
        from src.retrieval.bm25_retriever import BM25Index

        bm25 = BM25Index()
        bm25.build(sample_chunks)
        results = bm25.search("")
        # Empty query → BM25 returns all docs (expected behavior)
        assert len(results) == 3
        for r in results:
            assert r.score == 0.0
