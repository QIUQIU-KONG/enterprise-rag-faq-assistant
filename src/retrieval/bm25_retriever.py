"""
BM25 keyword search index using rank_bm25.
Supports build, search, persist (pickle), and load operations.
"""
import pickle
import re
from pathlib import Path
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from loguru import logger

from src.ingestion.chunker import Chunk
from config.settings import settings


@dataclass
class BM25Result:
    chunk_id: str
    score: float
    domain: str
    sub_topic: str
    title: str
    content: str
    source_file: str


class BM25Index:
    """BM25 index wrapper with persistence support."""

    def __init__(self):
        self._index: BM25Okapi | None = None
        self._chunks: list[Chunk] = []
        self._id_to_chunk: dict[str, Chunk] = {}

    @property
    def is_built(self) -> bool:
        return self._index is not None

    def build(self, chunks: list[Chunk]):
        """Build BM25 index from chunks."""
        self._chunks = chunks
        self._id_to_chunk = {c.id: c for c in chunks}

        # Tokenize: split Chinese + English words
        tokenized = [_tokenize(c.content) for c in chunks]
        self._index = BM25Okapi(tokenized)
        logger.info(f"BM25 index built: {len(chunks)} documents")

    def search(self, query: str, top_k: int | None = None) -> list[BM25Result]:
        """Search BM25 index and return ranked results."""
        if not self.is_built:
            raise RuntimeError("BM25 index not built. Call build() first.")

        top_k = top_k or settings.BM25_TOP_K
        tokenized_query = _tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        # Get top-k indices sorted by score
        indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results: list[BM25Result] = []
        for idx in indices:
            chunk = self._chunks[idx]
            results.append(BM25Result(
                chunk_id=chunk.id,
                score=float(scores[idx]),
                domain=chunk.domain,
                sub_topic=chunk.sub_topic,
                title=chunk.title,
                content=chunk.content,
                source_file=chunk.source_file,
            ))

        return results

    def save(self, path: Path | None = None):
        """Persist BM25 index to pickle file."""
        path = path or settings.PROCESSED_DATA_DIR / "bm25_index.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": self._chunks,
            "id_to_chunk": self._id_to_chunk,
            "index": self._index,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"BM25 index saved to {path}")

    def load(self, path: Path | None = None):
        """Load BM25 index from pickle file.

        Security: only load pickle files generated locally by this project.
        Do NOT load .pkl files from untrusted sources.
        """
        path = path or settings.PROCESSED_DATA_DIR / "bm25_index.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._chunks = data["chunks"]
        self._id_to_chunk = data["id_to_chunk"]
        self._index = data["index"]
        logger.info(f"BM25 index loaded from {path}: {len(self._chunks)} docs")

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._id_to_chunk.get(chunk_id)


def _tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25 indexing.
    Handles both Chinese (character-level bigrams) and English (word-level).
    """
    tokens: list[str] = []

    # Extract Chinese character sequences
    chinese_pattern = re.compile(r"[一-鿿]+")
    english_pattern = re.compile(r"[a-zA-Z]+")

    # Chinese: use overlapping bigrams + individual chars
    for match in chinese_pattern.finditer(text):
        segment = match.group()
        # Bigrams
        for i in range(len(segment) - 1):
            tokens.append(segment[i:i + 2])
        # Single characters for short words
        if len(segment) <= 2:
            tokens.append(segment)

    # English: lowercase words
    for match in english_pattern.finditer(text):
        word = match.group().lower()
        if len(word) >= 2:
            tokens.append(word)

    # Numbers
    for match in re.finditer(r"\d+", text):
        tokens.append(match.group())

    return tokens
