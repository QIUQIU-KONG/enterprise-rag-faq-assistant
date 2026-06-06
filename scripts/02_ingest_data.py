#!/usr/bin/env python
"""Step 2: Run full data ingestion pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.ingestion.pipeline import IngestionPipeline

if __name__ == "__main__":
    logger.info("Starting data ingestion pipeline...")
    pipeline = IngestionPipeline()
    pipeline.run(force=True)
    logger.info("Ingestion complete! Ready to serve queries.")
