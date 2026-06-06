#!/usr/bin/env python
"""Step 5: Run RAGAS evaluation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.evaluation.test_set_generator import generate_test_set, save_test_set
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.reporter import generate_report
from src.core.rag_engine import get_rag_engine

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("RAG Evaluation Pipeline")
    logger.info("=" * 50)

    # Step 1: Generate/load test set
    logger.info("\n[1/4] Generating test set...")
    pairs = generate_test_set()
    save_test_set(pairs)
    logger.info(f"  Generated {len(pairs)} QA pairs")

    # Step 2: Initialize RAG engine
    logger.info("\n[2/4] Initializing RAG engine...")
    engine = get_rag_engine()
    engine.initialize()
    logger.info(f"  Engine ready. Index size: {engine.index_size()} chunks")

    # Step 3: Run evaluation
    logger.info("\n[3/4] Running evaluation...")
    evaluator = RAGEvaluator(engine)
    report = evaluator.evaluate()

    # Step 4: Generate reports
    logger.info("\n[4/4] Generating reports...")
    report_path = generate_report(report)

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    overall = report.overall
    print(f"Faithfulness:       {overall.get('faithfulness', 'N/A'):.3f}")
    print(f"Answer Relevance:   {overall.get('answer_relevance', 'N/A'):.3f}")
    print(f"Context Precision:  {overall.get('context_precision', 'N/A'):.3f}")
    print(f"Context Recall:     {overall.get('context_recall', 'N/A'):.3f}")
    print(f"\nClarification Accuracy: {report.clarifications.get('clarification_accuracy', 'N/A'):.1%}")
    print(f"\nFull report: {report_path}")
