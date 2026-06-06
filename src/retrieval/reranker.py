"""
Cross-encoder reranker for fine-grained relevance scoring.
Re-ranks top hybrid search results using BGE-reranker-base.
"""
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

from src.retrieval.rrf_fusion import HybridResult
from config.settings import settings


@dataclass
class RankedResult:
    """Final ranked result after reranking."""
    chunk_id: str
    content: str
    domain: str
    sub_topic: str
    title: str
    source_file: str
    relevance_score: float  # Cross-encoder score
    rrf_score: float  # Original RRF score


class Reranker:
    """
    Cross-encoder based reranker using BGE-reranker-base.
    Uses the local ModelScope cache to avoid downloading from HuggingFace.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        # Prefer local ModelScope cache
        self._local_path = str(
            Path.home() / ".cache/modelscope/hub/models/BAAI/bge-reranker-base"
        )
        self._model = None
        self._tokenizer = None
        self._device = None

    def _lazy_load(self):
        """Lazily load the cross-encoder on first use."""
        if self._model is not None:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        model_path = self._local_path if Path(self._local_path).exists() else self.model_name
        logger.info(f"Loading reranker model from: {model_path}")

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._model.eval()

        self._device = torch.device("cpu")
        self._model.to(self._device)

        logger.info(f"Reranker loaded on {self._device}")

    def rerank(
        self,
        query: str,
        results: list[HybridResult],
        top_k: int | None = None,
    ) -> list[RankedResult]:
        """
        Rerank hybrid search results using cross-encoder.
        Scores each (query, document) pair for fine-grained relevance.
        """
        top_k = top_k or settings.RERANK_TOP_K

        if not results:
            return []

        self._lazy_load()

        import torch

        # Build (query, document) pairs
        pairs = [(query, r.content[:512]) for r in results]  # Truncate long texts

        # Tokenize all pairs
        features = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self._device)

        # Score all pairs
        with torch.no_grad():
            scores = self._model(**features).logits.squeeze(-1)

        # Handle single-pair case
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)

        scores = scores.cpu().tolist()
        if not isinstance(scores, list):
            scores = [scores]

        # Build ranked results
        ranked = []
        for result, score in zip(results, scores):
            # Normalize score to 0-1 range using sigmoid
            import numpy as np
            normalized = float(1.0 / (1.0 + np.exp(-float(score))))

            ranked.append(RankedResult(
                chunk_id=result.chunk_id,
                content=result.content,
                domain=result.domain,
                sub_topic=result.sub_topic,
                title=result.title,
                source_file=result.source_file,
                relevance_score=normalized,
                rrf_score=result.rrf_score,
            ))

        # Sort by relevance score descending (higher = more relevant)
        ranked.sort(key=lambda x: x.relevance_score, reverse=True)
        return ranked[:top_k]
