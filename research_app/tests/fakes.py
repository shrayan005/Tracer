"""Test doubles: fake LLM providers that behave exactly as a test script tells them to."""

from app.domain.errors import ProviderError
from app.integrations.llm.providers import ProviderResponse


class FakeProvider:
    """
    Each call returns/raises the next item of `script`:
        FakeProvider("a", [ProviderRateLimited("a", "slow down"), "hello"])
    -> 1st call raises 429, 2nd call returns "hello".
    """

    def __init__(self, name: str, script: list, price_per_m: float | None = 1.0):
        self.name = name
        self.model = f"{name}-model"
        self.script = list(script)
        self.calls = 0
        self.price_per_m = price_per_m
        self.last_kwargs = None

    async def chat(self, messages, *, max_tokens, temperature, json_mode):
        self.calls += 1
        self.last_kwargs = {"messages": messages, "json_mode": json_mode}
        item = self.script.pop(0)
        if isinstance(item, ProviderError):
            raise item
        return ProviderResponse(text=item, finish_reason="stop",
                                input_tokens=10, output_tokens=5, thinking_tokens=2)

    async def stream(self, messages, *, max_tokens, temperature, json_mode):
        """The same script, streamed: a text comes word by word, then its usage. Two more kinds of item:
            ProviderError             -> raised before the first word (the gateway may retry or switch)
            ("hello wor", error)      -> those words, then the error (the stream broke mid-way)"""
        self.calls += 1
        self.last_kwargs = {"messages": messages, "json_mode": json_mode}
        item = self.script.pop(0)
        if isinstance(item, ProviderError):
            raise item
        text, error = item if isinstance(item, tuple) else (item, None)
        for i, word in enumerate(text.split(" ")):
            yield word if i == 0 else f" {word}"
        if error is not None:
            raise error
        yield ProviderResponse(text=text, finish_reason="stop", input_tokens=10, output_tokens=5, thinking_tokens=2)

    def cost(self, input_tokens, output_tokens):
        if self.price_per_m is None:
            return None
        return (input_tokens + output_tokens) / 1e6 * self.price_per_m
