"""
Abstract base adapter — defines the universal adapter interface.

All platform adapters (Feishu, OpenClaw, Streamlit, etc.) implement this
interface so the RAG engine can serve any frontend without knowing
platform-specific protocols.

Design:
  parse(raw) → engine.query(text) → format(response)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class UnifiedMessage:
    """Platform-agnostic incoming message."""
    platform: str
    user_id: str
    text: str
    session_id: str | None = None


@dataclass
class UnifiedResponse:
    """Platform-agnostic response from the RAG engine."""
    text: str
    action: str  # "answer" | "counter_question"
    sources: list[dict] = field(default_factory=list)
    options: list[dict] = field(default_factory=list)
    counter_question: str | None = None


class BaseAdapter(ABC):
    """
    Abstract adapter: parse platform input → query engine → format output.

    Subclasses only need to implement parse() and format().
    The process() template method orchestrates the full pipeline.
    """

    def __init__(self, engine):
        """
        Args:
            engine: A RAGEngine instance (injected, not imported globally).
        """
        self._engine = engine

    @abstractmethod
    def parse(self, raw_data: dict) -> UnifiedMessage:
        """Parse platform-specific raw message into UnifiedMessage."""
        ...

    @abstractmethod
    def format(self, response: UnifiedResponse) -> dict:
        """Format UnifiedResponse into platform-specific output dict."""
        ...

    def process(self, raw_data: dict) -> dict:
        """
        Full adapter pipeline: parse → query → format.

        Args:
            raw_data: Platform-specific raw request dictionary.

        Returns:
            Platform-specific formatted response dictionary.
        """
        msg = self.parse(raw_data)
        result = self._engine.query(msg.text)
        unified = UnifiedResponse(
            text=result.get("answer", ""),
            action=result.get("action", "answer"),
            sources=result.get("sources", []),
            options=result.get("options", []),
            counter_question=result.get("counter_question"),
        )
        return self.format(unified)
