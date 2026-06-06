"""
Chinese-aware text chunker using LangChain's RecursiveCharacterTextSplitter.
Splits markdown documents into overlapping chunks for embedding and retrieval.
"""
import hashlib
from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config.settings import settings
from src.ingestion.loader import Document, extract_keywords


@dataclass
class Chunk:
    """A single chunk of text with metadata."""
    id: str
    content: str
    domain: str
    sub_topic: str
    title: str
    source_file: str
    chunk_index: int
    total_chunks: int
    keywords: list[str] = field(default_factory=list)
    language: str = "zh"


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create a splitter optimized for Chinese text."""
    return RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",     # Paragraph breaks
            "\n",       # Line breaks
            "。",       # Chinese period
            "！",       # Chinese exclamation
            "？",       # Chinese question mark
            "；",       # Chinese semicolon
            "，",       # Chinese comma
            ".",        # English period
            " ",        # Space (last resort)
        ],
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )


def chunk_document(doc: Document) -> list[Chunk]:
    """
    Split a single document into overlapping chunks.
    Each chunk inherits the document's metadata.
    """
    splitter = create_text_splitter()
    text_chunks = splitter.split_text(doc.content)

    chunks: list[Chunk] = []
    total = len(text_chunks)

    for i, text in enumerate(text_chunks):
        # Generate deterministic chunk ID
        chunk_id = _make_chunk_id(doc.source_file, i)

        # Extract keywords from chunk content
        keywords = extract_keywords(text)

        chunk = Chunk(
            id=chunk_id,
            content=text.strip(),
            domain=doc.domain,
            sub_topic=doc.sub_topic,
            title=doc.title,
            source_file=doc.source_file,
            chunk_index=i,
            total_chunks=total,
            keywords=keywords,
            language="zh",
        )
        chunks.append(chunk)

    logger.debug(
        f"Chunked {doc.source_file}: {total} chunks "
        f"({settings.CHUNK_SIZE} char, {settings.CHUNK_OVERLAP} overlap)"
    )
    return chunks


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    """Chunk all documents and return flat list of chunks."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))

    logger.info(
        f"Chunked {len(docs)} docs → {len(all_chunks)} chunks total"
    )
    return all_chunks


def _make_chunk_id(source_file: str, chunk_index: int) -> str:
    """Generate a deterministic chunk ID from source file + index."""
    raw = f"{source_file}#{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]
