"""Unit tests for provider helpers (no network)."""

from types import SimpleNamespace

from app.config import Settings
from app.integrations.llm.providers import build_providers, normalize_usage


def usage(prompt, completion, total, reasoning=None):
    details = SimpleNamespace(reasoning_tokens=reasoning) if reasoning is not None else None
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total,
                           completion_tokens_details=details)


def test_thinking_inside_completion_tokens_groq_style():
    assert normalize_usage(usage(80, 121, 201, reasoning=108)) == (80, 13, 108)


def test_thinking_outside_completion_tokens_gemini_style():
    assert normalize_usage(usage(10, 48, 401)) == (10, 48, 343)


def test_no_thinking():
    assert normalize_usage(usage(20, 30, 50)) == (20, 30, 0)


def test_providers_without_keys_are_skipped():
    settings = Settings(database_url="postgresql+psycopg://x", groq_api_key="k1", gemini_api_key=None,
                        openrouter_api_key="k3", llm_provider_order=["gemini", "groq", "openrouter"])
    assert [p.name for p in build_providers(settings)] == ["groq", "openrouter"]


# ---------- streaming (SSE step 3): the real provider class, with a fake OpenAI client ----------

class FakeChunks:
    """What `create(stream=True)` returns: chunks to iterate, closable with `async with`."""

    def __init__(self, chunks, error=None):
        self.chunks, self.error, self.closed = chunks, error, False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True

    async def __aiter__(self):
        for c in self.chunks:
            yield c
        if self.error:
            raise self.error


def chunk(text=None, finish=None, usage_=None):
    choices = [SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=finish)] if text or finish else []
    return SimpleNamespace(choices=choices, usage=usage_)


def provider_with(chunks: FakeChunks):
    from app.integrations.llm.providers import OpenAICompatibleProvider, ProviderConfig
    p = OpenAICompatibleProvider(ProviderConfig(name="groq", base_url="http://x", api_key="k", model="m"), 5)
    sent = {}

    async def create(**kwargs):
        sent.update(kwargs)
        return chunks
    p._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return p, sent


async def test_provider_streams_text_then_one_response_with_the_usage():
    chunks = FakeChunks([chunk('{"sen'), chunk('tences": []}'), chunk(finish="stop"),
                         chunk(usage_=usage(80, 121, 201, reasoning=108))])     # usage comes last, no text
    p, sent = provider_with(chunks)
    items = [i async for i in p.stream([{"role": "user", "content": "q"}], max_tokens=100, temperature=0,
                                        json_mode=True)]
    assert items[:2] == ['{"sen', 'tences": []}']
    final = items[-1]
    assert final.text == '{"sentences": []}' and final.finish_reason == "stop"
    assert (final.input_tokens, final.output_tokens, final.thinking_tokens) == (80, 13, 108)
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}
    assert chunks.closed


async def test_provider_turns_a_broken_stream_into_a_retryable_error():
    import httpx
    import pytest

    from app.domain.errors import ProviderUnavailable
    p, _ = provider_with(FakeChunks([chunk("Creatine")], error=httpx.RemoteProtocolError("peer closed")))
    stream = p.stream([], max_tokens=10, temperature=None, json_mode=False)
    assert await anext(stream) == "Creatine"
    with pytest.raises(ProviderUnavailable, match="broke off"):
        await anext(stream)
