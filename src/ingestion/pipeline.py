"""
Complete ingestion pipeline: Load → Chunk → Embed → Store.
Orchestrates the full document processing workflow.
"""
import json
from pathlib import Path
from loguru import logger

from src.ingestion.loader import load_documents, Document
from src.ingestion.chunker import chunk_documents, Chunk
from src.embeddings.embedder import get_embedder
from src.retrieval.bm25_retriever import BM25Index
from src.retrieval.dense_retriever import VectorStore
from config.settings import settings


class IngestionPipeline:
    """Orchestrates the full data ingestion workflow."""

    def __init__(self):
        self.bm25 = BM25Index()
        self.vector_store = VectorStore()

    def run(self, raw_dir: Path | None = None, force: bool = False):
        """
        Execute full ingestion pipeline.
        1. Load markdown documents
        2. Chunk into overlapping segments
        3. Build BM25 index
        4. Embed and store in ChromaDB
        """
        raw_dir = raw_dir or settings.RAW_DATA_DIR

        if force:
            logger.warning("Force mode: resetting vector store")
            self.vector_store.delete_collection()

        # Step 1: Load
        logger.info("=" * 50)
        logger.info("Step 1/4: Loading documents...")
        docs = load_documents(raw_dir)
        if not docs:
            logger.error(f"No markdown files found in {raw_dir}")
            return

        # Step 2: Chunk
        logger.info("Step 2/4: Chunking documents...")
        chunks = chunk_documents(docs)

        # Step 3: Build BM25
        logger.info("Step 3/4: Building BM25 index...")
        self.bm25.build(chunks)
        self.bm25.save()

        # Step 4: Embed + Store
        logger.info("Step 4/4: Embedding and storing...")
        # Process in batches for memory efficiency
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.vector_store.add_chunks(batch)
            logger.debug(f"  Embedded batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")

        # Save everything
        self.vector_store.save()
        self._save_chunks(chunks)

        logger.info("=" * 50)
        logger.info(f"Ingestion complete: {len(docs)} docs → {len(chunks)} chunks")
        logger.info(f"  BM25 index: {settings.PROCESSED_DATA_DIR / 'bm25_index.pkl'}")
        logger.info(f"  Vector store: {Path(settings.VECTOR_PERSIST_DIR) / 'vectors.pkl'}")
        logger.info(f"  Chunks: {settings.PROCESSED_DATA_DIR / 'chunks.json'}")

    def _save_chunks(self, chunks: list[Chunk]):
        """Save chunk metadata as JSON for reference."""
        settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        chunk_data = [
            {
                "id": c.id,
                "content_preview": c.content[:100],
                "domain": c.domain,
                "sub_topic": c.sub_topic,
                "title": c.title,
                "source_file": c.source_file,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                "keywords": c.keywords,
            }
            for c in chunks
        ]
        path = settings.PROCESSED_DATA_DIR / "chunks.json"
        path.write_text(json.dumps(chunk_data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Chunk metadata saved to {path}")
