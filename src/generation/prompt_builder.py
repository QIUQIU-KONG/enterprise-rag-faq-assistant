"""
Prompt templates for RAG-based FAQ answering.
Designed with the six-layer RAG prompt architecture:
  1. Role  2. Boundaries  3. Input format
  4. Output format  5. Anti-hallucination  6. Tone
"""
from jinja2 import Template
from src.knowledge_generator.domains import DOMAIN_LABELS, DOMAIN_SHORT_LABELS


SYSTEM_PROMPT = Template("""\
你是一个内部知识库助手。你的回答直接影响同事的出差安排和项目申报结果，因此必须准确、严谨。

## 核心规则（必须遵守）

1. **只回答问题**：直接回应用户的问题，不要客套话，不要说"你好"或"欢迎提问"。
2. **只使用提供的知识库**：不要使用你自身的知识。知识库里没有的信息，直接说"目前知识库中暂无相关信息，建议联系相关部门确认"。
3. **绝不会编造**：如果知识库只提供了部分信息，只回答有依据的部分，不要补全不存在的细节。
4. **必须标注来源**：每条回答末尾必须标注「📎 参考：《文档标题》」。多个来源用顿号分隔。

## 回答格式

按以下结构组织回答：

- **简单问题**（答案很明确）：直接给出答案，一两段即可。
- **清单类问题**（如问"需要什么材料"）：用分点列举，每点加粗关键词。
- **流程类问题**（如问"怎么办"）：用数字序号分步骤说明。
- **对比类问题**（如问"有什么区别"）：用表格呈现。

## 知识库内容

以下是你唯一可以使用的知识来源：

{% for ctx in contexts %}
---
📄 来源：{{ ctx.title }}
所属领域：{{ ctx.domain_label }}
内容：
{{ ctx.content }}
---
{% endfor %}

现在严格基于以上知识库内容回答用户问题。""")

USER_PROMPT = Template("""用户提问：{{ query }}""")

ANSWER_PROMPT = Template("""\
## 参考资料

{% for ctx in contexts %}
[{{ loop.index }}]《{{ ctx.title }}》({{ ctx.domain_label }})
{{ ctx.content }}

{% endfor %}
## 用户问题

{{ query }}

请根据上述参考资料回答。严格遵守：
- 只基于参考资料，不编造
- 答案中引用来源编号，如「[1]」
- 参考资料不充分时，指出缺口
- 使用中文""")

def build_rag_prompt(query: str, contexts: list[dict]) -> tuple[str, str]:
    """
    Build system and user prompts for RAG generation.
    Uses the six-layer architecture: Role → Boundaries → Input → Output → Anti-hallucination → Tone.

    Returns (system_prompt, user_prompt).
    """
    formatted_contexts = []
    for ctx in contexts:
        domain = ctx.get("domain", "")
        formatted_contexts.append({
            "title": ctx.get("title", "未知文档"),
            "domain_label": DOMAIN_LABELS.get(domain, domain),
            "content": ctx.get("content", ""),
        })

    system = SYSTEM_PROMPT.render(contexts=formatted_contexts)
    user = USER_PROMPT.render(query=query)
    return system, user


def build_answer_prompt(query: str, contexts: list[dict]) -> str:
    """Build a single-prompt version with numbered references."""
    formatted = []
    for ctx in contexts:
        domain = ctx.get("domain", "")
        formatted.append({
            "title": ctx.get("title", "未知"),
            "domain_label": DOMAIN_LABELS.get(domain, domain),
            "content": ctx.get("content", ""),
        })
    return ANSWER_PROMPT.render(query=query, contexts=formatted)


# TODO: LLM-based clarification for borderline cases.
# When the heuristic classifier is uncertain (domain ratio ≥80% but score gap < 0.12),
# an LLM call could provide better ambiguity judgment than the current fallback-to-clarify.
# Integration point: ClarificationEngine.decide() in src/clarification/decision_tree.py.
