"""
Counter-question templates for multi-domain ambiguous queries.
Provides natural Chinese clarification questions and option building.
All domain labels are imported from the single source of truth in domains.py.
"""
from src.knowledge_generator.domains import DOMAIN_LABELS, DOMAIN_SHORT_LABELS

# Counter-question templates
TWO_DOMAIN_TEMPLATE = (
    "您的问题「{query}」可能涉及以下两个方面：\n"
    "1️⃣ {domain_a}\n"
    "2️⃣ {domain_b}\n\n"
    "请问您想了解哪一个方向？"
)

THREE_DOMAIN_TEMPLATE = (
    "您的问题「{query}」涉及多个知识领域，请问您主要想了解：\n"
    "1️⃣ {domain_a}\n"
    "2️⃣ {domain_b}\n"
    "3️⃣ {domain_c}\n\n"
    "请告诉我您的侧重点，我会为您提供更精准的回答。"
)

VAGUE_TEMPLATE = (
    "抱歉，我暂时无法准确理解您的问题「{query}」。\n"
    "我的知识库覆盖以下领域：\n"
    "{options_list}\n"
    "您可以换一种说法，或者选择一个领域，我来帮您解答。"
)


def build_counter_question(query: str, domains: list[str]) -> str:
    """
    Build a natural counter-question based on detected domains.
    """
    domains = domains[:3]  # Max 3 domains
    labels = [DOMAIN_SHORT_LABELS.get(d, d) for d in domains]

    if len(domains) == 1:
        return (
            f"您的问题「{query}」我初步判断是关于{labels[0]}的，"
            f"但不太确定具体想问哪方面。能帮我再描述一下吗？"
        )

    if len(domains) == 2:
        return TWO_DOMAIN_TEMPLATE.format(
            query=query,
            domain_a=DOMAIN_LABELS.get(domains[0], domains[0]),
            domain_b=DOMAIN_LABELS.get(domains[1], domains[1]),
        )

    if len(domains) == 3:
        return THREE_DOMAIN_TEMPLATE.format(
            query=query,
            domain_a=DOMAIN_LABELS.get(domains[0], domains[0]),
            domain_b=DOMAIN_LABELS.get(domains[1], domains[1]),
            domain_c=DOMAIN_LABELS.get(domains[2], domains[2]),
        )

    # Vague — can't determine domains
    options_str = "\n".join(
        f"- {v}" for v in DOMAIN_LABELS.values()
    )
    return VAGUE_TEMPLATE.format(query=query, options_list=options_str)


def build_options(query: str, domains: list[str]) -> list[dict]:
    """
    Build structured options for UI buttons / Feishu card actions.
    Returns list of {label, domain, refined_query}.
    """
    options = []
    for d in domains[:3]:
        label = DOMAIN_SHORT_LABELS.get(d, d)
        options.append({
            "label": label,
            "domain": d,
            "refined_query": query,
        })
    return options
