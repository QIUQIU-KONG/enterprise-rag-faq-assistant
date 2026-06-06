"""
Embedding service using SentenceTransformer.
Provides batch encoding for efficient vector generation on CPU.
"""
from typing import overload
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from pathlib import Path
from config.settings import settings


class Embedder:
    """Wraps SentenceTransformer for batch encoding with caching."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model: SentenceTransformer | None = None
        self._dim = settings.EMBEDDING_DIM
        self._batch_size = settings.EMBEDDING_BATCH_SIZE

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            # Prefer local ModelScope cache if available
            model_path = self.model_name
            local_path = settings.EMBEDDING_MODEL_LOCAL
            if local_path and Path(local_path).exists():
                model_path = local_path
            logger.info(f"Loading embedding model: {model_path}")
            self._model = SentenceTransformer(model_path)
            logger.info(
                f"Model loaded (dim={self._model.get_sentence_embedding_dimension()})"
            )
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: str | list[str]) -> np.ndarray:
        """
        Embed a single text or list of texts.
        Returns numpy array of shape (n, dim) or (dim,).
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        if is_single:
            return embeddings[0]
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query (single text)."""
        return self.embed(query)

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        """Embed a list of document texts."""
        return self.embed(documents)


# Global embedder singleton
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Get or create the global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
