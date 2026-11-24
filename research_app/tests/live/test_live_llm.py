"""
Live test: real providers through the real gateway (costs a few tokens).
Skipped by default; run with:  pytest -m live
"""

import pytest
from pydantic import BaseModel

from app.dependencies import close_resources, get_llm_gateway

pytestmark = pytest.mark.live


@pytest.fixture(autouse=True)
async def cleanup():
    """Each test has its own event loop: close shared DB/Redis connections after each one."""
    yield
    await close_resources()


class Capital(BaseModel):
    country: str
    capital: str


async def test_real_text_completion():
    result = await get_llm_gateway().complete([{"role": "user", "content": "Reply with the word: ok"}],
                                              purpose="live_test", max_tokens=300)
    assert result.text and result.provider in get_llm_gateway().provider_names


async def test_real_structured_output():
    capital, result = await get_llm_gateway().complete_structured(
        [{"role": "user", "content": "What is the capital of Nepal?"}], Capital, purpose="live_test")
    assert capital.capital.lower() == "kathmandu"


@pytest.mark.parametrize("name", ["groq", "gemini", "openrouter"])
async def test_real_stream_gives_pieces_and_usage(name):
    """SSE step 3: does this provider stream, and send token counts at the end (stream_options)?"""
    provider = next((p for p in get_llm_gateway().providers if p.name == name), None)
    if provider is None:
        pytest.skip(f"{name} has no API key")
    items = [i async for i in provider.stream([{"role": "user", "content": "Count from 1 to 20, comma separated."}],
                                              max_tokens=300, temperature=0, json_mode=False)]
    pieces, final = items[:-1], items[-1]
    print(f"\n{name}: {len(pieces)} pieces, tokens in/out/thinking = "
          f"{final.input_tokens}/{final.output_tokens}/{final.thinking_tokens}, text: {final.text[:60]!r}")
    assert pieces and final.text and final.input_tokens > 0
