"""
LLM providers: one class that talks to any OpenAI-compatible API
(Groq, Gemini's OpenAI endpoint, OpenRouter, ...).

Its job is to make ONE request and translate the result:
  - success     -> ProviderResponse (text + normalised token counts)
  - any failure -> one of our ProviderError types (rate limited / unavailable / rejected)
stream() is the same request with stream=True: the text arrives in small pieces as the model writes it
(SSE step 3), and the token counts come last.
Retries and fallback are NOT done here; that is the gateway's job.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol

import httpx
import openai
from openai import AsyncOpenAI

from app.config import Settings
from app.domain.errors import ProviderError, ProviderRateLimited, ProviderRejected, ProviderUnavailable


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    price_input_per_m: float | None = None    # USD per 1M input tokens (None = unknown)
    price_output_per_m: float | None = None   # USD per 1M output tokens (thinking billed as output)
    extra_params: dict = field(default_factory=dict)


@dataclass
class ProviderResponse:
    text: str
    finish_reason: str | None
    input_tokens: int
    output_tokens: int      # visible answer
    thinking_tokens: int    # hidden reasoning


class ChatProvider(Protocol):
    """Anything with this shape can be used by the gateway (real providers and test fakes)."""
    name: str
    model: str

    async def chat(self, messages: list[dict], *, max_tokens: int, temperature: float | None,
                   json_mode: bool) -> ProviderResponse: ...

    def stream(self, messages: list[dict], *, max_tokens: int, temperature: float | None,
               json_mode: bool) -> AsyncIterator["str | ProviderResponse"]:
        """Text pieces (str) as they are written, then ONE ProviderResponse: the whole text and the usage."""
        ...

    def cost(self, input_tokens: int, output_tokens: int) -> float | None: ...


def normalize_usage(usage) -> tuple[int, int, int]:
    """
    Return (input, visible_output, thinking). Providers report thinking differently:
      Groq / OpenRouter: inside completion_tokens (completion_tokens_details.reasoning_tokens)
      Gemini:            outside completion_tokens, visible only in total_tokens
    """
    if usage is None:
        return 0, 0, 0
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) or 0
    if reasoning:
        return usage.prompt_tokens, usage.completion_tokens - reasoning, reasoning
    hidden = max((usage.total_tokens or 0) - usage.prompt_tokens - usage.completion_tokens, 0)
    return usage.prompt_tokens, usage.completion_tokens, hidden


def translate(name: str, error: openai.OpenAIError) -> ProviderError:
    """An OpenAI SDK error -> ours, so the gateway knows: wait and retry, or move to the next provider."""
    if isinstance(error, openai.RateLimitError):
        return ProviderRateLimited(name, _short(error), 429, _retry_after_seconds(error))
    if isinstance(error, openai.APIConnectionError):         # includes timeouts
        return ProviderUnavailable(name, f"connection/timeout: {_short(error)}")
    if isinstance(error, openai.APIStatusError):
        if error.status_code >= 500:
            return ProviderUnavailable(name, _short(error), error.status_code)
        return ProviderRejected(name, _short(error), error.status_code)
    return ProviderUnavailable(name, _short(error))           # e.g. a stream that broke off mid-way


def _retry_after_seconds(error: openai.APIStatusError) -> float | None:
    try:
        value = error.response.headers.get("retry-after")
        return float(value) if value else None
    except (AttributeError, ValueError):
        return None


class OpenAICompatibleProvider:
    def __init__(self, config: ProviderConfig, timeout_s: float):
        self.config = config
        self.name = config.name
        self.model = config.model
        # max_retries=0: the SDK must not retry on its own; the gateway decides.
        # AsyncOpenAI: its requests are awaited, so the event loop stays free while we wait.
        self._client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key,
                                   timeout=timeout_s, max_retries=0)

    def _request(self, messages, max_tokens, temperature, json_mode) -> dict:
        kwargs = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        kwargs.update(self.config.extra_params)
        return kwargs

    async def chat(self, messages, *, max_tokens, temperature, json_mode) -> ProviderResponse:
        try:
            response = await self._client.chat.completions.create(
                **self._request(messages, max_tokens, temperature, json_mode))
        except openai.OpenAIError as e:
            raise translate(self.name, e) from e

        if not response.choices:
            raise ProviderUnavailable(self.name, "empty response (no choices)")
        choice = response.choices[0]
        input_t, output_t, thinking_t = normalize_usage(response.usage)
        return ProviderResponse(
            text=(choice.message.content or "").strip(),
            finish_reason=choice.finish_reason,
            input_tokens=input_t, output_tokens=output_t, thinking_tokens=thinking_t,
        )

    async def stream(self, messages, *, max_tokens, temperature, json_mode) -> AsyncIterator[str | ProviderResponse]:
        # include_usage: the LAST chunk then carries the token counts (it has no text), so streamed
        # answers still count toward the user's budget
        request = self._request(messages, max_tokens, temperature, json_mode)
        pieces: list[str] = []
        finish_reason = None
        usage = None
        try:
            chunks = await self._client.chat.completions.create(
                **request, stream=True, stream_options={"include_usage": True})
            async with chunks:                            # closes the connection if the reader stops early
                async for chunk in chunks:
                    if chunk.usage is not None:
                        usage = chunk.usage
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    finish_reason = choice.finish_reason or finish_reason
                    text = choice.delta.content if choice.delta else None
                    if text:
                        pieces.append(text)
                        yield text
        except openai.OpenAIError as e:
            raise translate(self.name, e) from e
        except httpx.HTTPError as e:                      # the connection broke while reading the stream
            raise ProviderUnavailable(self.name, f"stream broke off: {_short(e)}") from e
        input_t, output_t, thinking_t = normalize_usage(usage)
        yield ProviderResponse(text="".join(pieces).strip(), finish_reason=finish_reason,
                               input_tokens=input_t, output_tokens=output_t, thinking_tokens=thinking_t)

    async def aclose(self) -> None:
        """Close the HTTP connections this client keeps open (they belong to one event loop)."""
        await self._client.close()

    def cost(self, input_tokens: int, output_tokens: int) -> float | None:
        if self.config.price_input_per_m is None or self.config.price_output_per_m is None:
            return None
        return round(input_tokens / 1e6 * self.config.price_input_per_m
                     + output_tokens / 1e6 * self.config.price_output_per_m, 6)


def _short(error: Exception, limit: int = 300) -> str:
    return str(error)[:limit]


# Known providers. Prices are USD per 1M tokens (check providers' pricing pages; None = unknown).
_PROVIDER_TEMPLATES = {
    "groq": dict(base_url="https://api.groq.com/openai/v1", key="groq_api_key", model="groq_model",
                 price_in=0.15, price_out=0.60,             # gpt-oss-120b on Groq
                 extra={"reasoning_effort": "low"}),         # measured: ~60% fewer thinking tokens
    # gpt-oss-120b on Groq
    "gemini": dict(base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                   key="gemini_api_key", model="gemini_model", price_in=None, price_out=None,
                   extra={"reasoning_effort": "low"}),      # stops thinking from eating max_tokens
    "openrouter": dict(base_url="https://openrouter.ai/api/v1", key="openrouter_api_key",
                       model="openrouter_model", price_in=0.0, price_out=0.0),   # ":free" model
}


def build_providers(settings: Settings) -> list[OpenAICompatibleProvider]:
    """Create providers in the configured order, skipping any without an API key."""
    providers = []
    for name in settings.llm_provider_order:
        template = _PROVIDER_TEMPLATES.get(name)
        api_key = getattr(settings, template["key"]) if template else None
        if not template or not api_key:
            continue
        config = ProviderConfig(
            name=name, base_url=template["base_url"], api_key=api_key,
            model=getattr(settings, template["model"]),
            price_input_per_m=template["price_in"], price_output_per_m=template["price_out"],
            extra_params=template.get("extra", {}),
        )
        providers.append(OpenAICompatibleProvider(config, settings.llm_timeout_seconds))
    return providers
