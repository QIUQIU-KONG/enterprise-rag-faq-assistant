"""
OpenClaw platform adapter.

Lightweight webhook adapter for OpenClaw bot integration.
Parses incoming webhook JSON, queries the RAG engine, and formats
a response compatible with OpenClaw's expected format.
"""
from loguru import logger

from src.adapters.base_adapter import BaseAdapter, UnifiedMessage, UnifiedResponse


class OpenClawAdapter(BaseAdapter):
    """OpenClaw webhook adapter: parse webhook → engine → format response."""

    def __init__(self, engine):
        super().__init__(engine)
        self.name = "FAQ Knowledge Base Bot"

    def parse(self, raw_data: dict) -> UnifiedMessage:
        """Parse OpenClaw webhook payload into UnifiedMessage."""
        return UnifiedMessage(
            platform="openclaw",
            user_id=raw_data.get("user_id", "anonymous"),
            text=raw_data.get("message", raw_data.get("text", "")),
            session_id=raw_data.get("session_id"),
        )

    def format(self, response: UnifiedResponse) -> dict:
        """Format UnifiedResponse for OpenClaw webhook response."""
        text = response.text
        if response.sources:
            text += "\n\n📎 参考来源:"
            for i, src in enumerate(response.sources[:3], 1):
                text += f"\n{i}. {src.get('title', '未知')}"

        result: dict = {"text": text, "action": response.action}
        if response.options:
            result["options"] = response.options
        if response.counter_question:
            result["counter_question"] = response.counter_question
        return result

    def handle_message(self, text: str, user_id: str = "anonymous") -> dict:
        """
        Direct message handling (convenience wrapper around process()).
        Handles input validation and empty messages gracefully.
        """
        if not text or not text.strip():
            return {
                "text": "您好！请问有什么可以帮助您的？\n\n"
                        "我可以回答关于出差注意事项、马来西亚商务签证、"
                        "项目申报材料方面的问题。",
                "action": "answer",
            }

        logger.info(f"OpenClaw message from {user_id}: {text[:100]}")
        return self.process({
            "message": text,
            "user_id": user_id,
        })
