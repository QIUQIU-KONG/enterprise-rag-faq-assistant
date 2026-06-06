"""
Markdown document loader with YAML frontmatter support.

Loads .md files from the raw data directory, parsing YAML frontmatter
for self-describing metadata (domain, title, description, etc.).
Falls back to directory-based inference for files without frontmatter.

Frontmatter format:
---
domain: travel_tips
subtopic: packing
title: "物品准备清单"
description: "出差马来西亚需要携带的证件、衣物、电子设备等物品"
last_updated: "2026-06-06"
status: reviewed
---
"""
from pathlib import Path
from dataclasses import dataclass, field
import re
from loguru import logger


@dataclass
class Document:
    """A loaded document before chunking."""
    content: str
    domain: str
    sub_topic: str
    title: str
    source_file: str
    metadata: dict = field(default_factory=dict)


def load_documents(raw_dir: Path) -> list[Document]:
    """
    Recursively load all markdown files from raw_dir.
    Domain and subtopic are read from YAML frontmatter when available,
    falling back to directory-name / filename inference.
    """
    docs: list[Document] = []

    for md_file in raw_dir.rglob("*.md"):
        raw_content = md_file.read_text(encoding="utf-8")

        # Parse YAML frontmatter
        frontmatter, body = _parse_frontmatter(raw_content)

        # Determine metadata — frontmatter wins, directory inference as fallback
        domain = frontmatter.get("domain", md_file.parent.name)
        sub_topic = frontmatter.get("subtopic", md_file.stem)
        title = frontmatter.get("title") or _extract_title(body)

        source_file = str(md_file.relative_to(raw_dir.parent))

        doc = Document(
            content=body.strip(),
            domain=domain,
            sub_topic=sub_topic,
            title=title,
            source_file=source_file,
            metadata={
                **frontmatter,
                "source_file": source_file,
            },
        )
        docs.append(doc)
        logger.debug(
            f"Loaded: {doc.source_file} → domain={domain}, "
            f"title={title} [frontmatter={'yes' if frontmatter else 'no'}]"
        )

    logger.info(f"Loaded {len(docs)} documents from {raw_dir}")
    return docs


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """
    Parse YAML-like frontmatter from markdown content.

    Returns (metadata_dict, body_content).
    For files without frontmatter, returns ({}, original_content).
    """
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    meta_text = parts[1].strip()
    body = parts[2]

    meta: dict[str, str] = {}
    for line in meta_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            # Strip quotes and whitespace from value
            value = value.strip().strip('"').strip("'")
            if value:
                meta[key] = value

    return meta, body


def _extract_title(content: str) -> str:
    """Extract the first H1 heading as document title."""
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    return match.group(1).strip() if match else "Untitled"


def extract_keywords(text: str, top_n: int = 5) -> list[str]:
    """
    Extract keywords from text using simple TF-based approach.
    A lightweight alternative to full NLP keyword extraction.
    """
    # Remove markdown syntax and common stopwords
    cleaned = re.sub(r"[#*>\-\|\`\[\]\(\)]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned)

    stopwords = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
        "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
        "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    }

    # Simple character-level n-gram extraction for Chinese
    words = []
    for match in re.finditer(r"[一-鿿]{2,}|[a-zA-Z]{3,}", cleaned):
        word = match.group().lower()
        if word not in stopwords:
            words.append(word)

    # Count frequency
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    # Return top N by frequency
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]
