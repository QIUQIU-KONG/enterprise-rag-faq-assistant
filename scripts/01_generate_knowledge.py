#!/usr/bin/env python
"""
Step 1: Scaffold knowledge documents and validate the knowledge base.

This script:
  1. Creates template .md files for any domain/subtopic that doesn't have one yet
  2. Runs validation to check all documents are complete
  3. Never overwrites existing content (safe to run anytime)

To add a new knowledge domain:
  1. Add domain + subtopics to src/knowledge_generator/domains.py
  2. Run this script to create template .md files
  3. Edit the .md files directly in data/raw/<domain>/
  4. Run scripts/02_ingest_data.py to rebuild indexes

To update existing content:
  Just edit the .md files directly, then run scripts/02_ingest_data.py.
  You do NOT need to run this script again.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.knowledge_generator.generator import (
    scaffold_knowledge_docs,
    validate_knowledge_docs,
    print_validation_report,
)

if __name__ == "__main__":
    # Step 1: Scaffold any missing template files
    logger.info("Scanning for missing knowledge documents...")
    created = scaffold_knowledge_docs(force=False)

    if created:
        logger.info(f"Created {len(created)} new template file(s):")
        for f in sorted(created):
            print(f"  [NEW] {f}")
        print("\n[WARN] Please edit the new template files with actual content, then re-run this script.")
    else:
        logger.info("All knowledge documents exist. Nothing to scaffold.")

    # Step 2: Validate
    print()
    report = validate_knowledge_docs()
    print_validation_report(report)

    if report["missing"] or report["empty"]:
        print("[TIP] Missing docs: run 'python scripts/01_generate_knowledge.py' to create templates")
        print("[TIP] Empty docs:  edit .md files directly in data/raw/<domain>/")
        print("[TIP] After editing: run 'python scripts/02_ingest_data.py' to rebuild indexes")
