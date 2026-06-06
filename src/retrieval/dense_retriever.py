"""
Vector store using numpy for dense embedding search.
Simple, cross-platform, no external dependencies beyond numpy.
"""
import pickle
from pathlib import Path
import numpy as np
from loguru import logger

from src.ingestion.chunker import Chunk
from src.embeddings.embedder import get_embedder
from config.settings import settings


class VectorStore:
    """Numpy-backed vector store for dense retrieval with cosine similarity."""

    def __init__(self):
        self._embedder = get_embedder()
        self._embeddings: np.ndarray | None = None  # shape: (n, dim)
        self._chunks: list[dict] = []  # chunk metadata
        self._ids: list[str] = []
        self._save_path = Path(settings.VECTOR_PERSIST_DIR) / "vectors.pkl"

    def add_chunks(self, chunks: list[Chunk]):
        """Add chunks to the vector store with embeddings."""
        if not chunks:
            return

        texts = [c.content for c in chunks]
        new_embeddings = self._embedder.embed_documents(texts)

        new_metas = [
            {
                "domain": c.domain,
                "sub_topic": c.sub_topic,
                "title": c.title,
                "source_file": c.source_file,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                "keywords": ",".join(c.keywords),
                "language": c.language,
                "content": c.content,
            }
            for c in chunks
        ]

        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        self._ids.extend([c.id for c in chunks])
        self._chunks.extend(new_metas)

        logger.info(f"Added {len(chunks)} chunks to vector store (total: {len(self._ids)})")

    def search(
        self,
        query: str,
        top_k: int | None = None,
        domain_filter: str | None = None,
    ) -> list[dict]:
        """
        Search vector store with optional domain filter using cosine similarity.
        Returns list of {id, content, metadata, score}.
        """
        top_k = top_k or settings.DENSE_TOP_K

        if self._embeddings is None or len(self._chunks) == 0:
            return []

        query_embedding = self._embedder.embed_query(query)
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Cosine similarity = dot product of normalized vectors
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-8
        normalized = self._embeddings / norms
        similarities = np.dot(normalized, query_norm)

        # Build (index, similarity) pairs with optional domain filter
        candidates = []
        for i, sim in enumerate(similarities):
            if domain_filter and self._chunks[i].get("domain") != domain_filter:
                continue
            candidates.append((i, float(sim)))

        # Sort by similarity descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:top_k]

        output = []
        for idx, sim in top:
            meta = self._chunks[idx]
            output.append({
                "id": self._ids[idx],
                "content": meta["content"],
                "metadata": meta,
                "score": sim,
            })

        return output

    def count(self) -> int:
        """Return total number of chunks in the store."""
        return len(self._ids)

    def get_by_domain(self, domain: str) -> list[dict]:
        """Retrieve all chunks for a specific domain."""
        output = []
        for i, meta in enumerate(self._chunks):
            if meta.get("domain") == domain:
                output.append({
                    "id": self._ids[i],
                    "content": meta["content"],
                    "metadata": meta,
                })
        return output

    def delete_collection(self):
        """Reset the vector store."""
        self._embeddings = None
        self._chunks = []
        self._ids = []
        logger.info("Vector store reset")

    def save(self):
        """Persist vector store to disk."""
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "embeddings": self._embeddings,
            "chunks": self._chunks,
            "ids": self._ids,
        }
        with open(self._save_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Vector store saved to {self._save_path}")

    def load(self):
        """Load vector store from disk.

        Security: only load pickle files generated locally by this project.
        Do NOT load .pkl files from untrusted sources.
        """
        if self._save_path.exists():
            with open(self._save_path, "rb") as f:
                data = pickle.load(f)
            self._embeddings = data["embeddings"]
            self._chunks = data["chunks"]
            self._ids = data["ids"]
            logger.info(f"Vector store loaded: {len(self._ids)} chunks")
        else:
            logger.warning(f"Vector store not found at {self._save_path}")
