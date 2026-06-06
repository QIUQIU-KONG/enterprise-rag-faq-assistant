"""
Pydantic request/response schemas for the FAQ API.
"""
from pydantic import BaseModel, Field


# --- Request schemas ---

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    domain: str | None = Field(None, description="Optional domain filter")
    history: list[dict] = Field(default_factory=list, description="Chat history")
    session_id: str | None = Field(None, description="Session identifier")


class ClarifySelectRequest(BaseModel):
    query: str = Field(..., description="Original query")
    selected_domain: str = Field(..., description="User-selected domain")
    session_id: str | None = Field(None)


# --- Response schemas ---

class SourceInfo(BaseModel):
    title: str
    domain: str
    content_snippet: str = Field(..., description="First 200 chars of source")
    source_file: str
    relevance_score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo] = Field(default_factory=list)
    action: str = "answer"  # "answer" | "counter_question"
    counter_question: str | None = None
    options: list[dict] = Field(default_factory=list)
    model: str = ""
    domain: str | None = None


class BotMessage(BaseModel):
    """Unified incoming bot message format."""
    platform: str  # "feishu" | "openclaw" | "web"
    user_id: str
    text: str
    timestamp: str | None = None
    extra: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    version: str
    index_size: int
