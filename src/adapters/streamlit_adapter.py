"""
Streamlit adapter — wraps the RAG engine for direct local calls.

Unlike Feishu/OpenClaw adapters which handle webhook protocols,
this adapter provides a lightweight wrapper used by app/streamlit_app.py
to call the engine directly (no HTTP overhead in development).

Also supports domain-filtered queries and clarification refinement.
"""
from src.adapters.base_adapter import BaseAdapter, UnifiedMessage, UnifiedResponse


class StreamlitAdapter(BaseAdapter):
    """Local adapter for Streamlit UI: wraps engine for direct Python calls."""

    def parse(self, raw_data: dict) -> UnifiedMessage:
        """Parse Streamlit form input into UnifiedMessage."""
        return UnifiedMessage(
            platform="web",
            user_id=raw_data.get("user_id", "web_user"),
            text=raw_data.get("query", raw_data.get("text", "")),
            session_id=raw_data.get("session_id"),
        )

    def format(self, response: UnifiedResponse) -> dict:
        """Format UnifiedResponse for Streamlit UI consumption."""
        return {
            "text": response.text,
            "action": response.action,
            "sources": response.sources,
            "options": response.options,
            "counter_question": response.counter_question,
        }

    def process_with_domain(
        self,
        query: str,
        domain_filter: str | None = None,
    ) -> dict:
        """
        Process a query with optional domain filter.

        Extends the base process() to support the engine's domain_filter
        parameter, which narrows search to a specific knowledge domain.
        """
        result = self._engine.query(query, domain_filter=domain_filter)
        unified = UnifiedResponse(
            text=result.get("answer", ""),
            action=result.get("action", "answer"),
            sources=result.get("sources", []),
            options=result.get("options", []),
            counter_question=result.get("counter_question"),
        )
        return self.format(unified)

    def process_clarify(self, query: str, selected_domain: str) -> dict:
        """
        Handle a user's domain selection from a counter-question.
        Refines the original query with domain context and re-queries.
        """
        refined_query = self._engine.clarification.refine_query(
            query, selected_domain
        )
        return self.process_with_domain(refined_query, domain_filter=selected_domain)
