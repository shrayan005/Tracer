"""
Embeddings (step 7.6a): text -> a list of numbers whose closeness means "similar meaning".

Behind a small Embedder interface, so the model can be swapped (the index remembers which model made each
vector). Gemini's REST API, batches of 100. Gemini returns vectors that are NOT unit length at 768
dimensions (measured: length 0.59), so they are normalised here: cosine distance then behaves.

Task types: Gemini embeds a passage and a question differently on purpose (RETRIEVAL_DOCUMENT /
RETRIEVAL_QUERY), which makes question -> passage search better than plain similarity.
Every request is logged like an LLM call (purpose "embed"), so it counts toward the user's budget.

Free-tier limit (measured, Google's own 429 message): 100 embedded TEXTS per minute per model, and a batch of
100 counts as 100. So the embedder paces itself with a per-minute budget shared by every embedder of the
same model in this process (90/minute: a margin), and on a 429 waits as long as Google says (RetryInfo).
"""

import asyncio
import math
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Literal, Protocol

import httpx

from app.common.logging import get_logger
from app.domain.errors import ProviderError, ProviderRateLimited, ProviderRejected, ProviderUnavailable
from app.domain.schemas import LlmCallRecord
from app.integrations.llm.run_context import current_run_id, current_user_id

log = get_logger(__name__)
Task = Literal["document", "query"]
TASK_TYPES = {"document": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"}
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
BATCH = 100                      # the API's limit per request
MAX_WAIT = 60.0                  # never wait longer than this for a retry: fail fast instead
UsageSink = Callable[[LlmCallRecord], Awaitable[None]]


class MinuteBudget:
    """At most `per_minute` texts in any 60 seconds: waits BEFORE sending instead of hitting the limit."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic):
        self.per_minute, self.clock = per_minute, clock
        self.sent: deque[tuple[float, int]] = deque()   # (when, how many texts)
        self.lock = asyncio.Lock()

    async def take(self, n: int, sleep: Callable[[float], Awaitable[None]]) -> None:
        async with self.lock:                          # one sender at a time, so the count stays right
            while True:
                now = self.clock()
                while self.sent and now - self.sent[0][0] >= 60:
                    self.sent.popleft()
                if sum(k for _, k in self.sent) + n <= self.per_minute:
                    self.sent.append((now, n))
                    return
                await sleep(60 - (now - self.sent[0][0]) + 0.5)


_BUDGETS: dict[str, MinuteBudget] = {}                # per model: shared by every embedder in this process


def budget_for(model: str, per_minute: int) -> MinuteBudget:
    if model not in _BUDGETS:
        _BUDGETS[model] = MinuteBudget(per_minute)
    return _BUDGETS[model]


class Embedder(Protocol):
    name: str                    # e.g. "gemini-embedding-001:768": stored with every vector
    dimensions: int

    async def embed(self, texts: list[str], task: Task) -> list[list[float]]: ...


def normalise(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(v * v for v in vector))
    return [v / length for v in vector] if length else vector


class GeminiEmbedder:
    def __init__(self, client: httpx.AsyncClient, api_key: str, model: str = "gemini-embedding-001",
                 dimensions: int = 768, usage_sink: UsageSink | None = None, max_retries: int = 4,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep, per_minute: int = 90,
                 budget: MinuteBudget | None = None):
        self.client, self.api_key, self.model, self.dimensions = client, api_key, model, dimensions
        self.name = f"{model}:{dimensions}"
        self.usage_sink, self.max_retries, self._sleep = usage_sink, max_retries, sleep
        self.batch = min(BATCH, per_minute)
        self.budget = budget or budget_for(model, per_minute)

    async def embed(self, texts: list[str], task: Task) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch):
            vectors += await self._batch(texts[start:start + self.batch], task)
        return vectors

    async def _batch(self, texts: list[str], task: Task) -> list[list[float]]:
        body = {"requests": [{"model": f"models/{self.model}", "content": {"parts": [{"text": t}]},
                              "taskType": TASK_TYPES[task], "outputDimensionality": self.dimensions}
                             for t in texts]}
        for attempt in range(1, self.max_retries + 2):
            await self.budget.take(len(texts), self._sleep)     # pace: never more than the free tier allows
            started = time.perf_counter()
            try:
                vectors = await self._post(body, len(texts))
            except ProviderError as error:
                await self._record(texts, attempt, started, error)
                if not error.retryable or attempt > self.max_retries:
                    raise
                wait = error.retry_after or min(60.0, 5.0 * 2 ** attempt)
                if wait > MAX_WAIT:      # measured (7.7): an unbounded wait hung a run for 15+ minutes
                    raise ProviderRateLimited("gemini-embed", f"Google asks to wait {wait:.0f}s (quota used up?)",
                                              status=429, retry_after=wait) from error
                log.warning("embedding: %s, retry %d in %.0fs", error, attempt, wait)
                await self._sleep(wait)
                continue
            await self._record(texts, attempt, started)
            return [normalise(v) for v in vectors]
        raise AssertionError("unreachable")

    async def _post(self, body: dict, expected: int) -> list[list[float]]:
        try:
            response = await self.client.post(URL.format(model=self.model), json=body,
                                              headers={"x-goog-api-key": self.api_key}, timeout=60)
        except httpx.HTTPError as e:
            raise ProviderUnavailable("gemini-embed", f"network error: {e}") from e
        if response.status_code == 429:
            raise ProviderRateLimited("gemini-embed", "rate limited", status=429, retry_after=retry_delay(response))
        if response.status_code >= 500:
            raise ProviderUnavailable("gemini-embed", f"HTTP {response.status_code}", status=response.status_code)
        if response.status_code >= 400:
            raise ProviderRejected("gemini-embed", f"HTTP {response.status_code}: {response.text[:200]}",
                                   status=response.status_code)
        vectors = [e.get("values") or [] for e in response.json().get("embeddings", [])]
        if len(vectors) != expected or any(len(v) != self.dimensions for v in vectors):
            raise ProviderUnavailable("gemini-embed", f"expected {expected} vectors of {self.dimensions} numbers")
        return vectors

    async def _record(self, texts: list[str], attempt: int, started: float, error: ProviderError | None = None):
        if self.usage_sink is None:
            return
        tokens = sum(len(t) for t in texts) // 4         # the API doesn't report tokens: ~4 characters per token
        await self.usage_sink(LlmCallRecord(
            purpose="embed", provider="gemini", model=self.model, status="error" if error else "ok",
            attempt=attempt, input_tokens=0 if error else tokens,
            latency_ms=int((time.perf_counter() - started) * 1000), cost_usd=None,
            error=str(error)[:500] if error else None, run_id=current_run_id(), user_id=current_user_id()))


def retry_delay(response: httpx.Response) -> float | None:
    """How long Google asks us to wait: the Retry-After header, or RetryInfo.retryDelay ("25s") in the body."""
    header = response.headers.get("retry-after")
    if header and header.isdigit():
        return float(header)
    try:
        for detail in response.json().get("error", {}).get("details", []):
            if detail.get("@type", "").endswith("RetryInfo"):
                match = re.fullmatch(r"([\d.]+)s", str(detail.get("retryDelay", "")))
                if match:
                    return float(match.group(1)) + 1
    except ValueError:
        pass
    return None
