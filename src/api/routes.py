"""
FastAPI route handlers — platform-agnostic /chat API only.

All platform-specific webhook handling (Feishu, OpenClaw, etc.)
is delegated to adapters in src/adapters/.
"""
import time
from fastapi import APIRouter, Header, HTTPException, Depends
from loguru import logger

from src.api.schemas import (
    ChatRequest, ChatResponse, ClarifySelectRequest,
    HealthResponse, SourceInfo,
)
from src.core.rag_engine import get_rag_engine
from config.settings import settings

router = APIRouter()


def verify_api_key(x_api_key: str | None = Header(default=None)):
    """Optional API key check. If API_KEY is configured, requests must provide it."""
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def _safe_chat_response(engine, query: str, domain_filter: str | None = None) -> dict:
    """Wrap engine.query() with error handling and timing."""
    t0 = time.time()
    try:
        result = engine.query(query, domain_filter=domain_filter)
        elapsed = time.time() - t0
        logger.info(f"Query completed in {elapsed:.2f}s: action={result.get('action')}")
        return result
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"Query failed in {elapsed:.2f}s: {type(e).__name__}")
        return {
            "answer": "系统处理请求时出现错误，请稍后重试。如问题持续，请联系管理员。",
            "sources": [],
            "action": "answer",
            "counter_question": None,
            "options": [],
            "domain": None,
            "model": "error",
        }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        engine = get_rag_engine()
        return HealthResponse(
            status="ok",
            version="1.0.0",
            index_size=engine.index_size(),
        )
    except Exception:
        return HealthResponse(status="degraded", version="1.0.0", index_size=0)


@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, _: bool = Depends(verify_api_key)):
    """
    Main chat endpoint — platform-agnostic.

    Handles query → clarify/answer → response.
    Supports optional domain pre-selection to skip clarification.
    """
    engine = get_rag_engine()

    query = request.query
    domain_filter = None
    if request.domain:
        query = engine.clarification.refine_query(request.query, request.domain)
        domain_filter = request.domain  # Skip clarification when domain is pre-selected

    result = _safe_chat_response(engine, query, domain_filter=domain_filter)

    sources = [
        SourceInfo(
            title=s["title"],
            domain=s["domain"],
            content_snippet=s["content"][:200],
            source_file=s["source_file"],
            relevance_score=s["score"],
        )
        for s in result.get("sources", [])
    ]

    return ChatResponse(
        answer=result.get("answer", ""),
        sources=sources,
        action=result.get("action", "answer"),
        counter_question=result.get("counter_question"),
        options=result.get("options", []),
        model=result.get("model", ""),
        domain=result.get("domain"),
    )


@router.post("/api/chat/clarify", response_model=ChatResponse)
async def clarify_select(request: ClarifySelectRequest, _: bool = Depends(verify_api_key)):
    """
    Handle user's domain selection from a counter-question.
    Refines the query with the selected domain and re-runs the pipeline.
    """
    engine = get_rag_engine()

    refined_query = engine.clarification.refine_query(
        request.query, request.selected_domain
    )

    result = _safe_chat_response(engine, refined_query, domain_filter=request.selected_domain)

    sources = [
        SourceInfo(
            title=s["title"],
            domain=s["domain"],
            content_snippet=s["content"][:200],
            source_file=s["source_file"],
            relevance_score=s["score"],
        )
        for s in result.get("sources", [])
    ]

    return ChatResponse(
        answer=result.get("answer", ""),
        sources=sources,
        action="answer",
        model=result.get("model", ""),
        domain=request.selected_domain,
    )
