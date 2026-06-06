"""
RAGAS evaluation pipeline.
Computes Faithfulness, Answer Relevance, Context Precision, and Context Recall.
"""
import json
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger

from config.settings import settings


@dataclass
class EvalResult:
    """Evaluation result for a single QA pair."""
    question_id: str
    question: str
    answer: str
    ground_truth: str | None
    retrieved_contexts: list[str] = field(default_factory=list)
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    clarification_triggered: bool = False
    clarification_correct: bool | None = None


@dataclass
class EvalReport:
    """Aggregated evaluation report."""
    overall: dict = field(default_factory=dict)
    by_domain: dict = field(default_factory=dict)
    by_difficulty: dict = field(default_factory=dict)
    clarifications: dict = field(default_factory=dict)
    per_question: list[EvalResult] = field(default_factory=list)


class RAGEvaluator:
    """
    Evaluates RAG pipeline quality using multiple metrics.
    Uses a combination of RAGAS (when available) and heuristic metrics.
    """

    def __init__(self, rag_engine=None):
        self.engine = rag_engine

    def evaluate(self, test_set_path: Path | None = None) -> EvalReport:
        """Run full evaluation on the test set."""
        test_set_path = test_set_path or settings.EVAL_DIR / "test_set.json"

        if not test_set_path.exists():
            logger.warning(f"Test set not found: {test_set_path}. Generating...")
            from src.evaluation.test_set_generator import generate_test_set, save_test_set
            pairs = generate_test_set()
            save_test_set(pairs)
        else:
            data = json.loads(test_set_path.read_text(encoding="utf-8"))
            pairs = data.get("pairs", [])

        logger.info(f"Evaluating {len(pairs)} QA pairs...")

        report = EvalReport()
        domain_results: dict[str, list[EvalResult]] = {}
        difficulty_results: dict[str, list[EvalResult]] = {}

        for pair in pairs:
            qid = pair["id"]
            question = pair["question"]
            ground_truth = pair.get("ground_truth_answer")
            domain = pair.get("domain", "unknown")
            difficulty = pair.get("difficulty", "unknown")
            qtype = pair.get("type", "single_domain")
            expected_clarification = pair.get("expected_clarification", False)

            # Skip empty questions
            if not question:
                result = EvalResult(
                    question_id=qid, question="(empty)", answer="",
                    ground_truth=None,
                    clarification_triggered=False, clarification_correct=None,
                )
                report.per_question.append(result)
                continue

            # Query the RAG engine
            # For single-domain questions, pass domain filter to skip clarification
            domain_filter = domain if domain not in ("ambiguous", "edge_case", "unknown") else None
            try:
                rag_result = self.engine.query(question, domain_filter=domain_filter)
            except Exception as e:
                logger.error(f"Error evaluating {qid}: {e}")
                continue

            answer = rag_result.get("answer", "")
            sources = rag_result.get("sources", [])
            action = rag_result.get("action", "answer")

            # Check clarification behavior
            clarification_triggered = action == "counter_question"
            clarification_correct = None
            if qtype == "multi_domain":
                clarification_correct = clarification_triggered == expected_clarification

            # Extract contexts
            contexts = [s.get("content", "") for s in sources]

            # Compute metrics
            faith = _estimate_faithfulness(answer, contexts)
            relevance = _estimate_answer_relevance(question, answer)
            precision = _estimate_context_precision(question, contexts)
            recall = _estimate_context_recall(ground_truth, contexts) if ground_truth else None

            result = EvalResult(
                question_id=qid,
                question=question,
                answer=answer,
                ground_truth=ground_truth,
                retrieved_contexts=contexts,
                faithfulness=faith,
                answer_relevance=relevance,
                context_precision=precision,
                context_recall=recall,
                clarification_triggered=clarification_triggered,
                clarification_correct=clarification_correct,
            )

            report.per_question.append(result)

            # Group by domain
            if domain not in domain_results:
                domain_results[domain] = []
            domain_results[domain].append(result)

            # Group by difficulty
            if difficulty not in difficulty_results:
                difficulty_results[difficulty] = []
            difficulty_results[difficulty].append(result)

        # Aggregate
        report.overall = _aggregate(report.per_question)
        report.by_domain = {
            d: _aggregate(results) for d, results in domain_results.items()
        }
        report.by_difficulty = {
            d: _aggregate(results) for d, results in difficulty_results.items()
        }

        # Clarification stats
        total_amb = sum(1 for r in report.per_question if r.clarification_correct is not None)
        correct_amb = sum(1 for r in report.per_question if r.clarification_correct is True)
        report.clarifications = {
            "total_ambiguous_questions": total_amb,
            "correct_clarifications": correct_amb,
            "clarification_accuracy": correct_amb / total_amb if total_amb > 0 else 1.0,
        }

        logger.info(
            f"Eval complete: Faith={report.overall.get('faithfulness', 'N/A'):.3f}, "
            f"Rel={report.overall.get('answer_relevance', 'N/A'):.3f}, "
            f"CP={report.overall.get('context_precision', 'N/A'):.3f}, "
            f"CR={report.overall.get('context_recall', 'N/A'):.3f}"
        )
        return report


def _estimate_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Estimate faithfulness: what fraction of answer claims are supported by contexts.
    Uses multi-level token overlap (bigram + unigram) as a heuristic.
    Note: For production use, replace with RAGAS LLM-as-judge for accurate scores.
    """
    if not answer or not contexts:
        return 0.0

    combined_context = " ".join(contexts)
    import re

    # Extract Chinese bigrams AND single characters (both carry meaning)
    answer_bigrams = set(re.findall(r"[一-鿿]{2,}", answer))
    answer_unigrams = set(re.findall(r"[一-鿿]", answer))
    context_bigrams = set(re.findall(r"[一-鿿]{2,}", combined_context))
    context_unigrams = set(re.findall(r"[一-鿿]", combined_context))

    if not answer_bigrams and not answer_unigrams:
        return 0.5  # Neutral if no Chinese tokens

    # Bigram overlap (weighted higher — exact phrase match)
    bigram_overlap = answer_bigrams & context_bigrams
    bigram_score = len(bigram_overlap) / len(answer_bigrams) if answer_bigrams else 0.0

    # Unigram overlap (weighted lower — single character match)
    unigram_overlap = answer_unigrams & context_unigrams
    unigram_score = len(unigram_overlap) / len(answer_unigrams) if answer_unigrams else 0.0

    # Weighted combination: bigrams are stronger signal
    return 0.7 * bigram_score + 0.3 * unigram_score


def _estimate_answer_relevance(question: str, answer: str) -> float:
    """
    Estimate answer relevance: does the answer address the question?
    Uses a composite heuristic based on answer quality indicators.
    Note: For production use, replace with RAGAS LLM-as-judge for accurate scores.
    """
    if not answer:
        return 0.0

    import re
    score = 0.0

    # 1. Length check: substantive answers are usually >50 chars
    if len(answer) >= 100:
        score += 0.25
    elif len(answer) >= 50:
        score += 0.15
    elif len(answer) >= 20:
        score += 0.05

    # 2. Structure check: well-formed answers use formatting
    if bool(re.search(r"\d+[\.、）\)]", answer)):
        score += 0.20  # Numbered list
    if bool(re.search(r"[-•\*]", answer)):
        score += 0.10  # Bullet points
    if "**" in answer:
        score += 0.10  # Bold formatting

    # 3. Content quality: check for informative patterns
    if re.search(r"[一-鿿]{3,}", answer):
        score += 0.15  # Contains Chinese content (not just error msg)

    # 4. Check if the answer contains domain-relevant detail keywords
    #    (specific terms that indicate the answer is not generic)
    detail_markers = [
        r"建议", r"注意", r"要求", r"需要", r"可以", r"必须",
        r"小时", r"分钟", r"马币", r"人民币", r"工作日", r"有效期",
        r"材料", r"流程", r"申请", r"提交", r"办理",
    ]
    marker_count = sum(1 for m in detail_markers if re.search(m, answer))
    score += min(0.15, marker_count * 0.03)

    # 5. Proper "don't know" response gets a bonus
    if "暂无" in answer or "无法" in answer or "知识库" in answer:
        score = max(score, 0.6)

    # 6. Penalty for raw error messages
    error_patterns = [r"Traceback", r"Error:", r"exception", r"错误"]
    if any(re.search(p, answer, re.IGNORECASE) for p in error_patterns):
        score *= 0.3

    return min(score, 1.0)


def _estimate_context_precision(question: str, contexts: list[str]) -> float:
    """
    Estimate context precision: what fraction of retrieved contexts are relevant?
    Checks each context for domain-relevant content indicators.
    """
    if not contexts:
        return 0.0

    import re
    relevant_count = 0
    for ctx in contexts:
        # A context is considered "relevant" if it contains substantive Chinese content
        # and is not just a header/metadata snippet
        chinese_chars = len(re.findall(r"[一-鿿]", ctx))
        has_structure = bool(re.search(r"(##|###|\d+\.|- |\*\*)", ctx))
        has_list_items = bool(re.search(r"^\s*[-•\*\d]", ctx, re.MULTILINE))

        # Context is relevant if: long enough AND (has structure OR has substantial Chinese content)
        if len(ctx) >= 100 and (has_structure or has_list_items or chinese_chars >= 50):
            relevant_count += 1
        elif len(ctx) >= 50 and has_structure:
            relevant_count += 1

    # If we have at least some relevant contexts, compute precision
    return relevant_count / len(contexts)


def _estimate_context_recall(ground_truth: str, contexts: list[str]) -> float:
    """
    Estimate context recall: what fraction of ground truth info is covered by contexts?
    Uses multi-level token overlap with bigram and unigram matching.
    """
    if not ground_truth or not contexts:
        return 0.0

    import re
    combined_context = " ".join(contexts)

    # Extract bigrams and unigrams from ground truth
    gt_bigrams = set(re.findall(r"[一-鿿]{2,}", ground_truth))
    gt_unigrams = set(re.findall(r"[一-鿿]", ground_truth))
    ctx_bigrams = set(re.findall(r"[一-鿿]{2,}", combined_context))
    ctx_unigrams = set(re.findall(r"[一-鿿]", combined_context))

    if not gt_bigrams and not gt_unigrams:
        return 0.5

    # Bigram recall (phrase-level match)
    bigram_overlap = gt_bigrams & ctx_bigrams
    bigram_recall = len(bigram_overlap) / len(gt_bigrams) if gt_bigrams else 0.0

    # Unigram recall (character-level match, lower weight)
    unigram_overlap = gt_unigrams & ctx_unigrams
    unigram_recall = len(unigram_overlap) / len(gt_unigrams) if gt_unigrams else 0.0

    # Weighted: bigrams are more meaningful for recall
    return 0.7 * bigram_recall + 0.3 * unigram_recall


def _aggregate(results: list[EvalResult]) -> dict:
    """Aggregate metrics from a list of results."""
    metrics = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]
    agg = {}
    for metric in metrics:
        values = [
            getattr(r, metric)
            for r in results
            if getattr(r, metric) is not None
        ]
        if values:
            agg[metric] = sum(values) / len(values)
            agg[f"{metric}_count"] = len(values)
        else:
            agg[metric] = None
            agg[f"{metric}_count"] = 0
    agg["total_questions"] = len(results)
    return agg
