"""
Evaluation report generator.
Produces markdown and JSON reports from evaluation results.
"""
import json
from pathlib import Path
from datetime import datetime
from loguru import logger

from src.evaluation.evaluator import EvalReport, EvalResult
from config.settings import settings


def generate_report(report: EvalReport) -> Path:
    """Generate evaluation report in markdown and JSON formats."""
    settings.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON report
    json_path = settings.EVAL_RESULTS_DIR / f"metrics_{timestamp}.json"
    _write_json_report(report, json_path)

    # Markdown report
    md_path = settings.EVAL_RESULTS_DIR / f"report_{timestamp}.md"
    _write_md_report(report, md_path)

    # Detailed CSV
    csv_path = settings.EVAL_RESULTS_DIR / f"per_question_{timestamp}.csv"
    _write_csv_report(report, csv_path)

    logger.info(f"Reports generated: {json_path}, {md_path}, {csv_path}")
    return md_path


def _write_json_report(report: EvalReport, path: Path):
    """Write JSON report."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "overall": report.overall,
        "by_domain": report.by_domain,
        "by_difficulty": report.by_difficulty,
        "clarifications": report.clarifications,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_md_report(report: EvalReport, path: Path):
    """Write markdown report."""
    lines = [
        "# RAG 评估报告",
        "",
        f"**评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**测试集大小**: {report.overall.get('total_questions', 0)} 条",
        "",
        "---",
        "",
        "## 1. 总体指标",
        "",
        "| 指标 | 分数 | 说明 |",
        "|------|------|------|",
    ]

    metrics = {
        "faithfulness": "忠实度 (Faithfulness)",
        "answer_relevance": "答案相关性 (Answer Relevance)",
        "context_precision": "检索精度 (Context Precision)",
        "context_recall": "检索召回率 (Context Recall)",
    }

    for key, name in metrics.items():
        value = report.overall.get(key)
        if value is not None:
            bar = _progress_bar(value)
            lines.append(f"| {name} | {value:.3f} {bar} | {_metric_desc(key)} |")
        else:
            lines.append(f"| {name} | N/A | {_metric_desc(key)} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. 按领域分析",
        "",
        "| 领域 | Faithfulness | Relevance | Precision | Recall |",
        "|------|-------------|-----------|-----------|--------|",
    ])

    domain_names = {
        "travel_tips": "出差注意事项",
        "malaysia_visa": "马来西亚商务签证",
        "project_applications": "项目申报材料",
        "ambiguous": "多域模糊查询",
        "edge_case": "边界测试",
    }

    for domain, metrics_data in report.by_domain.items():
        label = domain_names.get(domain, domain)
        f = _fmt(metrics_data.get("faithfulness"))
        r = _fmt(metrics_data.get("answer_relevance"))
        p = _fmt(metrics_data.get("context_precision"))
        rec = _fmt(metrics_data.get("context_recall"))
        lines.append(f"| {label} | {f} | {r} | {p} | {rec} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. 按难度分析",
        "",
        "| 难度 | Faithfulness | Relevance | Precision | Recall |",
        "|------|-------------|-----------|-----------|--------|",
    ])

    diff_labels = {
        "easy": "简单",
        "medium": "中等",
        "hard": "困难",
        "ambiguous": "模糊(需反问)",
        "edge": "边界/异常",
    }

    for diff, metrics_data in report.by_difficulty.items():
        label = diff_labels.get(diff, diff)
        f = _fmt(metrics_data.get("faithfulness"))
        r = _fmt(metrics_data.get("answer_relevance"))
        p = _fmt(metrics_data.get("context_precision"))
        rec = _fmt(metrics_data.get("context_recall"))
        lines.append(f"| {label} | {f} | {r} | {p} | {rec} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. 反问澄清效果",
        "",
        f"- 模糊问题总数: {report.clarifications.get('total_ambiguous_questions', 0)}",
        f"- 正确触发反问: {report.clarifications.get('correct_clarifications', 0)}",
        f"- 反问准确率: {report.clarifications.get('clarification_accuracy', 0):.1%}",
        "",
        "---",
        "",
        "## 5. 问题级别详情 (Top 10 Best & Worst)",
        "",
        "### 最佳表现",
        "",
    ])

    # Best by faithfulness
    valid_results = [r for r in report.per_question if r.faithfulness is not None]
    best = sorted(valid_results, key=lambda x: x.faithfulness or 0, reverse=True)[:5]

    for r in best:
        lines.append(f"- **{r.question}** → Faith: {r.faithfulness:.3f}, Rel: {r.answer_relevance or 0:.3f}")

    lines.extend([
        "",
        "### 最需改进",
        "",
    ])

    worst = sorted(valid_results, key=lambda x: x.faithfulness or 0)[:5]
    for r in worst:
        lines.append(f"- **{r.question}** → Faith: {r.faithfulness:.3f}, Rel: {r.answer_relevance or 0:.3f}")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv_report(report: EvalReport, path: Path):
    """Write per-question details as CSV."""
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "question", "answer", "faithfulness",
                         "answer_relevance", "context_precision", "context_recall",
                         "clarification_triggered", "clarification_correct"])
        for r in report.per_question:
            writer.writerow([
                r.question_id, r.question, r.answer[:200],
                r.faithfulness, r.answer_relevance, r.context_precision,
                r.context_recall, r.clarification_triggered, r.clarification_correct,
            ])


def _progress_bar(value: float) -> str:
    """ASCII progress bar for markdown."""
    bar_len = 10
    filled = int(value * bar_len)
    return f"`[{'█' * filled}{'░' * (bar_len - filled)}]`"


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def _metric_desc(key: str) -> str:
    """Short description of each metric."""
    descriptions = {
        "faithfulness": "回答内容是否可溯源到知识库",
        "answer_relevance": "回答是否直接回应问题",
        "context_precision": "检索结果中相关文档的排位",
        "context_recall": "该检索到的文档是否都检索到了",
    }
    return descriptions.get(key, "")
