"""
LLM provider abstraction supporting Anthropic and OpenAI.
Retry logic lives here so it's consistent whether you use providers
directly or through guarded_call.
"""

import os
from typing import Optional

import anthropic as _anthropic_mod

try:
    import openai as _openai_mod
    _OPENAI_AVAILABLE = True
except ImportError:
    _openai_mod = None  # type: ignore[assignment]
    _OPENAI_AVAILABLE = False

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception


def _is_rate_limit(exc: BaseException) -> bool:
    """Return True for HTTP 429 / rate-limit errors from any provider SDK."""
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg or "too many requests" in msg


def _retry():
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True,
    )


def _retry_with_429():
    """Retry on any error (including 429s) with exponential backoff."""
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=16),
        reraise=True,
    )


class AnthropicProvider:
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = _anthropic_mod.Anthropic(api_key=key)
        self.async_client = _anthropic_mod.AsyncAnthropic(api_key=key)

    @_retry()
    def complete(self, prompt: str, model: str, max_tokens: int) -> str:
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def acomplete(self, prompt: str, model: str, max_tokens: int) -> str:
        from tenacity import AsyncRetrying
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=8),
            reraise=True,
        ):
            with attempt:
                response = await self.async_client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text


class OpenAIProvider:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        if not _OPENAI_AVAILABLE:
            raise ImportError("openai package required: pip install openai")
        key = api_key or os.getenv("OPENAI_API_KEY")
        url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.client = _openai_mod.OpenAI(api_key=key, base_url=url)
        self.async_client = _openai_mod.AsyncOpenAI(api_key=key, base_url=url)

    @_retry_with_429()
    def complete(self, prompt: str, model: str, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def acomplete(self, prompt: str, model: str, max_tokens: int) -> str:
        from tenacity import AsyncRetrying
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=2, min=2, max=16),
            reraise=True,
        ):
            with attempt:
                response = await self.async_client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content


class GeminiProvider(OpenAIProvider):
    """
    Google Gemini via its OpenAI-compatible endpoint.
    Uses GEMINI_API_KEY env var (or api_key argument).
    Model examples: gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro
    """
    _GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("GEMINI_API_KEY")
        super().__init__(api_key=key, base_url=self._GEMINI_BASE_URL)


def get_provider(
    name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> "AnthropicProvider | OpenAIProvider | GeminiProvider":
    """
    Factory. Reads PROVIDER env var (default 'anthropic').
    Pass name to override the env var.
    """
    resolved = (name or os.getenv("PROVIDER", "anthropic")).lower()
    if resolved == "anthropic":
        return AnthropicProvider(api_key=api_key)
    elif resolved == "openai":
        return OpenAIProvider(api_key=api_key)
    elif resolved == "gemini":
        return GeminiProvider(api_key=api_key)
    raise ValueError(
        f"Unknown provider '{resolved}'. Supported: anthropic, openai, gemini"
    )
