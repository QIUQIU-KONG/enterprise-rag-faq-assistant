"""
Knowledge document scaffolding and validation tool.

All knowledge content is maintained as independent .md files in data/raw/.
This module:
  - Scaffolds template .md files for new domains/subtopics (never overwrites existing)
  - Validates that all expected files exist and have correct structure
  - Reports on knowledge base coverage and health

Content workflow:
  1. New domain → run scaffold_knowledge_docs() to create template .md files
  2. Edit .md files directly in data/raw/<domain>/<subtopic>.md
  3. Run validate_knowledge_docs() to check for issues
  4. Run scripts/02_ingest_data.py to rebuild indexes
"""
from pathlib import Path
from datetime import date
from loguru import logger

from src.knowledge_generator.domains import (
    Domain, DOMAIN_SUBTOPICS, DOMAIN_LABELS, DOMAIN_SHORT_LABELS,
)
from config.settings import settings

# YAML frontmatter template for new scaffolded files
FRONTMATTER_TEMPLATE = """\
---
domain: {domain}
subtopic: {subtopic_id}
title: "{title}"
description: "{description}"
last_updated: "{today}"
status: draft
---
"""


def scaffold_knowledge_docs(force: bool = False) -> list[Path]:
    """
    Create template .md files for any domain+subtopic that doesn't have one yet.
    Never overwrites existing files unless force=True.

    Returns list of created file paths (empty if nothing new).
    """
    created_files: list[Path] = []
    today = date.today().isoformat()

    for domain in Domain:
        domain_dir = settings.RAW_DATA_DIR / domain.value
        domain_dir.mkdir(parents=True, exist_ok=True)

        subtopics = DOMAIN_SUBTOPICS.get(domain.value, [])
        for subtopic in subtopics:
            filepath = domain_dir / f"{subtopic['id']}.md"

            if filepath.exists() and not force:
                logger.debug(f"Exists, skip: {filepath}")
                continue

            frontmatter = FRONTMATTER_TEMPLATE.format(
                domain=domain.value,
                subtopic_id=subtopic["id"],
                title=subtopic["title"],
                description=subtopic["description"],
                today=today,
            )
            body = _build_template_body(subtopic)
            content = frontmatter + "\n" + body

            filepath.write_text(content, encoding="utf-8")
            created_files.append(filepath)
            action = "Created" if not filepath.exists() or force else "Overwritten"
            logger.info(f"{action}: {filepath}")

    return created_files


def _build_template_body(subtopic: dict) -> str:
    """Build a minimal template body with the subtopic structure."""
    lines = [
        f"# {subtopic['title']}",
        "",
        f"> {subtopic['description']}",
        "",
    ]
    topics = subtopic.get("topics", [])
    if topics:
        for t in topics:
            lines.append(f"## {t}")
            lines.append("")
            lines.append("<!-- TODO: 填写内容 -->")
            lines.append("")
    else:
        lines.append("<!-- TODO: 填写内容 -->")
        lines.append("")
    return "\n".join(lines)


def validate_knowledge_docs() -> dict:
    """
    Validate all knowledge documents exist and have proper structure.
    Returns dict with validation results.
    """
    result = {
        "total_expected": 0,
        "existing": 0,
        "missing": [],
        "empty": [],
        "no_frontmatter": [],
        "by_domain": {},
    }

    for domain in Domain:
        domain_dir = settings.RAW_DATA_DIR / domain.value
        domain_label = DOMAIN_SHORT_LABELS.get(domain.value, domain.value)
        domain_stats = {"total": 0, "ok": 0, "issues": 0}
        subtopics = DOMAIN_SUBTOPICS.get(domain.value, [])

        for subtopic in subtopics:
            filepath = domain_dir / f"{subtopic['id']}.md"
            result["total_expected"] += 1
            domain_stats["total"] += 1

            if not filepath.exists():
                result["missing"].append(str(filepath))
                domain_stats["issues"] += 1
                continue

            content = filepath.read_text(encoding="utf-8")
            result["existing"] += 1

            # Check for frontmatter
            if not content.startswith("---"):
                result["no_frontmatter"].append(str(filepath))
                domain_stats["issues"] += 1

            # Check for empty content (only frontmatter or template placeholders)
            body = _strip_frontmatter(content)
            if not body or len(body.strip()) < 50:
                result["empty"].append(str(filepath))
                domain_stats["issues"] += 1
            else:
                domain_stats["ok"] += 1

        result["by_domain"][domain_label] = domain_stats

    return result


def print_validation_report(result: dict):
    """Print a human-readable validation report."""
    print("\n" + "=" * 60)
    print("Knowledge Base Validation Report")
    print("=" * 60)
    print(f"Expected docs:   {result['total_expected']}")
    print(f"Existing docs:   {result['existing']}")
    print(f"Missing:         {len(result['missing'])}")
    print(f"Empty/template:  {len(result['empty'])}")
    print(f"No frontmatter:  {len(result['no_frontmatter'])}")

    if result["missing"]:
        print("\n--- Missing Documents ---")
        for f in result["missing"]:
            print(f"  [MISS] {f}")

    if result["empty"]:
        print("\n--- Empty / Template-only Documents ---")
        for f in result["empty"]:
            print(f"  [EMPTY] {f}")

    if result["no_frontmatter"]:
        print("\n--- Missing YAML Frontmatter ---")
        for f in result["no_frontmatter"]:
            print(f"  [NO-META] {f}")

    print("\n--- By Domain ---")
    for domain, stats in result["by_domain"].items():
        bar = _health_bar(stats["ok"], stats["total"])
        print(f"  {domain}: {stats['ok']}/{stats['total']} OK {bar}")

    all_ok = not result["missing"] and not result["empty"]
    status = "PASS" if all_ok else "NEEDS FIX"
    print(f"\nOverall: {status}")
    print("=" * 60 + "\n")


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from content, returning the body."""
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return parts[2]
    return content


def _health_bar(ok: int, total: int) -> str:
    """Simple health bar for CLI output."""
    if total == 0:
        return ""
    ratio = ok / total
    filled = int(ratio * 10)
    return f"[{'█' * filled}{'░' * (10 - filled)}]"


# Backward-compatible alias for existing scripts
generate_knowledge_docs = scaffold_knowledge_docs
