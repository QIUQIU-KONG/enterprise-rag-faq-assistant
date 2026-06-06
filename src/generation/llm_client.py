"""
LLM client abstraction supporting multiple backends:
- Ollama (local, free, qwen2.5:3b)
- Groq (cloud, free tier, llama-3.1-8b-instant)
- DeepSeek (cloud, paid, deepseek-chat)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from loguru import logger

from config.settings import settings


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> LLMResponse:
        ...


class OllamaClient(LLMClient):
    """Ollama local LLM client."""

    def __init__(self):
        import ollama
        self._client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        self._model = settings.OLLAMA_MODEL

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> LLMResponse:
        try:
            response = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": kwargs.get("temperature", 0.3),
                    "num_predict": kwargs.get("max_tokens", 1024),
                },
            )
            content = response["message"]["content"]
            return LLMResponse(content=content, model=self._model, provider="ollama")
        except Exception as e:
            logger.error(f"Ollama call failed: {type(e).__name__}")
            raise


class GroqClient(LLMClient):
    """Groq cloud LLM client (free tier)."""

    def __init__(self):
        from groq import Groq
        self._client = Groq(api_key=settings.GROQ_API_KEY)
        self._model = settings.GROQ_MODEL

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> LLMResponse:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=kwargs.get("temperature", 0.3),
                max_tokens=kwargs.get("max_tokens", 1024),
            )
            content = response.choices[0].message.content or ""
            return LLMResponse(content=content, model=self._model, provider="groq")
        except Exception as e:
            logger.error(f"Groq call failed: {type(e).__name__}")
            raise


class DeepSeekClient(LLMClient):
    """DeepSeek cloud LLM client (paid, best Chinese quality)."""

    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
            timeout=30.0,  # 30-second timeout
            max_retries=2,  # Retry twice before giving up
        )
        self._model = settings.DEEPSEEK_MODEL

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> LLMResponse:
        """Generate with timeout and retry. Raises on failure for caller to handle."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=kwargs.get("temperature", 0.3),
                max_tokens=kwargs.get("max_tokens", 1024),
                timeout=30.0,
            )
            content = response.choices[0].message.content or ""
            return LLMResponse(content=content, model=self._model, provider="deepseek")
        except Exception as e:
            logger.error(f"DeepSeek API call failed: {type(e).__name__}")
            raise


class MockClient(LLMClient):
    """Mock LLM for testing without a real model."""

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> LLMResponse:
        return LLMResponse(
            content=f"[Mock LLM - 无实际模型]\n\n检索到相关上下文，但未配置LLM。请设置Ollama/Groq/DeepSeek。\n\n"
                    f"用户问题摘要: {user_prompt[:200]}...",
            model="mock",
            provider="debug",
        )


def create_llm_client() -> LLMClient:
    """
    Factory function to create the appropriate LLM client.
    Falls back from primary to secondary if primary is unavailable.
    """
    provider = settings.LLM_PROVIDER

    if provider == "debug":
        return MockClient()

    if provider == "ollama":
        try:
            return OllamaClient()
        except Exception as e:
            logger.warning(f"Ollama unavailable ({e}), trying Groq fallback...")
            provider = "groq"

    if provider == "groq":
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env file")
        return GroqClient()

    if provider == "deepseek":
        if not settings.DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY not set in .env file")
        return DeepSeekClient()

    raise ValueError(f"Unknown LLM provider: {provider}. Set LLM_PROVIDER in .env")
