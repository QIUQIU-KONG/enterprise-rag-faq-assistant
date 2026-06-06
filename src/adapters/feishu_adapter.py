"""
Feishu platform adapter.

Handles:
  - Feishu webhook message parsing (JSON event → UnifiedMessage)
  - Feishu interactive card formatting (UnifiedResponse → card JSON)
  - Feishu API integration (signature verification, token management, message sending)
"""
import hashlib
import hmac
import json
import time
import httpx
from loguru import logger

from src.adapters.base_adapter import BaseAdapter, UnifiedMessage, UnifiedResponse
from config.settings import settings


class FeishuAdapter(BaseAdapter):
    """Full Feishu platform adapter: webhook → engine → card response."""

    def __init__(self, engine):
        super().__init__(engine)
        self.app_id = settings.FEISHU_APP_ID
        self.app_secret = settings.FEISHU_APP_SECRET
        self.verification_token = settings.FEISHU_VERIFICATION_TOKEN
        self._tenant_access_token: str | None = None
        self._token_expires_at: float = 0.0

    # --- Adapter interface ---

    def parse(self, raw_data: dict) -> UnifiedMessage:
        """Parse Feishu event payload into UnifiedMessage."""
        event = raw_data.get("event", {})
        message = event.get("message", {})
        sender = event.get("sender", {})

        text = ""
        content = message.get("content", "{}")
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                pass
        if isinstance(content, dict):
            text = content.get("text", "")

        return UnifiedMessage(
            platform="feishu",
            user_id=sender.get("sender_id", "unknown"),
            text=text,
            session_id=message.get("chat_id"),
        )

    def format(self, response: UnifiedResponse) -> dict:
        """Build Feishu interactive card response from UnifiedResponse."""
        if response.action == "counter_question":
            return self._build_counter_question_card(response)
        return self._build_answer_card(response)

    # --- Card builders ---

    def _build_counter_question_card(self, response: UnifiedResponse) -> dict:
        """Build a card with option buttons for multi-domain clarification."""
        elements = [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": response.counter_question or response.text,
                },
            },
        ]

        actions = []
        for opt in response.options:
            actions.append({
                "tag": "button",
                "text": {"tag": "plain_text", "content": opt.get("label", "?")},
                "value": opt,
                "type": "primary",
            })

        if actions:
            elements.append({"tag": "action", "actions": actions})

        return {
            "msg_type": "interactive",
            "card": {
                "header": {"title": {"tag": "plain_text", "content": "AI 知识助手"}},
                "elements": elements,
            },
        }

    def _build_answer_card(self, response: UnifiedResponse) -> dict:
        """Build an answer card with source citations."""
        source_md = "\n\n---\n📎 **参考来源**\n"
        for i, src in enumerate(response.sources[:3], 1):
            source_md += f"\n{i}. {src.get('title', '未知')}"

        return {
            "msg_type": "interactive",
            "card": {
                "header": {"title": {"tag": "plain_text", "content": "AI 知识助手"}},
                "elements": [
                    {
                        "tag": "div",
                        "text": {"tag": "lark_md", "content": response.text},
                    },
                    {
                        "tag": "div",
                        "text": {"tag": "lark_md", "content": source_md},
                    },
                ],
            },
        }

    # --- Feishu API integration ---

    def verify_signature(self, timestamp: str, nonce: str, signature: str) -> bool:
        """Verify Feishu request signature (HMAC-SHA256 challenge)."""
        if not self.app_secret:
            logger.warning("FEISHU_APP_SECRET not configured, skipping signature verification")
            return True

        parts = sorted([timestamp, nonce, self.app_secret, signature])
        raw = "".join(parts).encode("utf-8")
        computed = hashlib.sha256(raw).hexdigest()
        return computed == signature

    async def get_tenant_access_token(self) -> str:
        """Get or refresh Feishu tenant access token."""
        if self._tenant_access_token and time.time() < self._token_expires_at - 300:
            return self._tenant_access_token

        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        payload = {"app_id": self.app_id, "app_secret": self.app_secret}

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload)
            data = resp.json()

        if data.get("code") != 0:
            raise Exception(f"Failed to get tenant access token: {data}")

        self._tenant_access_token = data["tenant_access_token"]
        self._token_expires_at = time.time() + data.get("expire", 7200)
        logger.info("Feishu tenant access token refreshed")
        return self._tenant_access_token

    async def send_message(
        self,
        open_id: str,
        response: UnifiedResponse,
        msg_type: str = "interactive",
    ):
        """Send a message to a Feishu user via the IM API."""
        token = await self.get_tenant_access_token()

        content = self.format(response)

        url = "https://open.feishu.cn/open-apis/im/v1/messages"
        params = {"receive_id_type": "open_id"}

        body = {
            "receive_id": open_id,
            "msg_type": content.get("msg_type", msg_type),
            "content": json.dumps(content.get("card", content)),
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, params=params, json=body)
            data = resp.json()

        if data.get("code") != 0:
            logger.error(f"Failed to send Feishu message: {data}")

        return data

    def build_card(
        self,
        title: str,
        content: str,
        sources: list[dict] | None = None,
        buttons: list[dict] | None = None,
    ) -> dict:
        """Build a Feishu card for manual message construction."""
        elements = [
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": content},
            },
        ]

        if sources:
            footer_text = "\n".join(
                f"• {s.get('title', 'Unknown')}" for s in sources[:5]
            )
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"\n---\n📎 **参考来源**\n{footer_text}",
                },
            })

        if buttons:
            actions = [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": b.get("label", "?")},
                    "value": b,
                    "type": "primary",
                }
                for b in buttons
            ]
            elements.append({"tag": "action", "actions": actions})

        return {
            "header": {"title": {"tag": "plain_text", "content": title}},
            "elements": elements,
        }
