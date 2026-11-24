"""
LLM gateway: the ONE entry point every service uses to talk to LLMs.

Reliability rules:
  1. Try providers in order (e.g. groq -> openrouter -> gemini).
  2. Temporary error (429, 5xx, timeout) -> wait (exponential backoff + jitter,
     or the provider's Retry-After) and retry the SAME provider, up to max_retries.
  3. Permanent error for that provider (bad key, unknown model, bad request)
     -> skip straight to the NEXT provider.
  4. Every attempt (ok or error) is reported to the usage sink -> llm_calls table.
  5. If every provider fails -> AllProvidersFailed.

stream() (SSE step 3) sends the text piece by piece as the model writes it. The same rules apply until
the FIRST piece arrives (retry, switch, fall back); after that the user is already reading this
provider's words, so a break is an error (StreamInterrupted), never a silent switch to another model.

complete_structured() adds: ask for JSON, validate with a Pydantic schema, and
if invalid, send the validation error back and ask again (learned in P9).

Switch instead of waiting (switch_after_s): if a provider asks us to wait longer than this
(e.g. Groq's free tier says "Retry-After: 18s") and another provider is still untried, try that one
now. The skipped provider is kept as a last resort, so a long wait never becomes a failure.

Routes (per-task provider order): `routes={"summarize": ["gemini"]}` makes summary calls try
gemini FIRST, then the normal order as fallback. Spreading tasks over providers spreads the load
over several free-tier rate limits (e.g. screening on groq, summaries on gemini).
"""

import asyncio
import json
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.common.logging import get_logger
from app.domain.errors import AllProvidersFailed, ProviderError, StreamInterrupted, StructuredOutputError
from app.domain.schemas import LlmCallRecord, LlmResult
from app.integrations.llm.providers import ChatProvider, ProviderResponse
from app.integrations.llm.run_context import current_run_id, current_user_id

log = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)
UsageSink = Callable[[LlmCallRecord], Awaitable[None]]   # async: saving to the DB is awaited
# one try against one provider: (provider, attempt number, start time) -> a result, or raises ProviderError
Attempt = Callable[[ChatProvider, int, float], Awaitable[Any]]


@dataclass
class _Call:
    """One complete() request, shared by every provider attempt (so attempts and errors add up)."""
    messages: list[dict]
    purpose: str
    max_tokens: int
    temperature: float | None
    json_mode: bool
    run_id: int | None
    attempts: int = 0
    errors: list[ProviderError] = field(default_factory=list)


class LlmGateway:
    def __init__(self, providers: list[ChatProvider], *, usage_sink: UsageSink | None = None,
                 max_retries: int = 2, backoff_base_s: float = 1.0, backoff_max_s: float = 20.0,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 routes: dict[str, list[str]] | None = None, switch_after_s: float | None = None,
                 wait_for: dict[str, float] | None = None):
        self.providers = providers
        self.routes = routes or {}     # purpose -> provider names to try first
        self.switch_after_s = switch_after_s   # asked to wait longer than this? try the next provider first
        # purpose -> seconds: tasks whose QUALITY depends on the routed model wait longer before switching
        # (measured 6.6: connections fell from 83% to ~30% recall when a busy groq was swapped for flash-lite)
        self.wait_for = wait_for or {}
        self.usage_sink = usage_sink
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self._sleep = sleep   # injectable so tests don't really wait

    @property
    def provider_names(self) -> list[str]:
        return [p.name for p in self.providers]

    def providers_for(self, purpose: str) -> list[ChatProvider]:
        """The route's providers first (only ones that exist), then all others in the normal order."""
        by_name = {p.name: p for p in self.providers}
        first = [by_name[name] for name in self.routes.get(purpose, []) if name in by_name]
        return first + [p for p in self.providers if p not in first]

    async def aclose(self) -> None:
        """Close every provider's HTTP connections (call when the app or CLI shuts down)."""
        for provider in self.providers:
            close = getattr(provider, "aclose", None)   # test fakes may not have one
            if close is not None:
                await close()

    # ---------- plain text ----------

    async def complete(self, messages: list[dict], *, purpose: str, max_tokens: int = 2000,
                       temperature: float | None = None, json_mode: bool = False,
                       run_id: int | None = None) -> LlmResult:
        call = self._call(messages, purpose, max_tokens, temperature, json_mode, run_id)

        async def once(provider: ChatProvider, attempt: int, start: float) -> LlmResult:
            response = await provider.chat(call.messages, max_tokens=call.max_tokens,
                                           temperature=call.temperature, json_mode=call.json_mode)
            return await self._finished(call, provider, attempt, start, response)

        return await self._run(call, once)

    # ---------- streamed text ----------

    async def stream(self, messages: list[dict], *, purpose: str, max_tokens: int = 2000,
                     temperature: float | None = None, json_mode: bool = False,
                     run_id: int | None = None) -> AsyncIterator[str | LlmResult]:
        """Text pieces (str) as the model writes them, then ONE LlmResult (the whole text, tokens, cost).
        Retries and fallback work as in complete() until the first piece arrives."""
        call = self._call(messages, purpose, max_tokens, temperature, json_mode, run_id)

        async def open_stream(provider: ChatProvider, attempt: int, start: float):
            pieces = provider.stream(call.messages, max_tokens=call.max_tokens,
                                     temperature=call.temperature, json_mode=call.json_mode)
            try:
                first = await anext(pieces)          # most failures (429, bad key, 5xx) happen before this
            except BaseException:
                await pieces.aclose()
                raise
            return provider, attempt, start, first, pieces

        provider, attempt, start, first, pieces = await self._run(call, open_stream)
        written: list[str] = []
        finished = False
        try:
            item = first
            while True:
                if isinstance(item, ProviderResponse):
                    finished = True
                    yield await self._finished(call, provider, attempt, start, item)
                    return
                written.append(item)
                yield item
                try:
                    item = await anext(pieces)
                except StopAsyncIteration:           # a provider must end with its ProviderResponse
                    raise StreamInterrupted(provider.name, "the stream ended without its usage") from None
                except ProviderError as error:
                    finished = True                  # recorded here, as an error
                    await self._record(call.purpose, provider, attempt, start, call.run_id, error=error)
                    raise StreamInterrupted(provider.name, str(error)) from error
        finally:
            await pieces.aclose()
            if not finished:
                # the reader stopped early (the user left): the model still worked, so it still counts.
                # The provider's own token count never came: estimate ~4 characters per token.
                await self._record_estimate(call, provider, attempt, start, "".join(written))

    def stream_json(self, messages: list[dict], schema: type[BaseModel], *, purpose: str,
                    max_tokens: int = 4000, temperature: float | None = 0) -> AsyncIterator[str | LlmResult]:
        """stream(), asking for JSON matching `schema` (as complete_structured does). The caller reads the
        pieces and validates the whole text at the end: a stream can't be re-asked once it has been shown."""
        return self.stream(_with_schema_instruction(messages, schema), purpose=purpose, max_tokens=max_tokens,
                           temperature=temperature, json_mode=True)

    # ---------- shared by complete() and stream() ----------

    def _call(self, messages, purpose, max_tokens, temperature, json_mode, run_id) -> "_Call":
        # an explicit run_id wins; otherwise the run scope we're in (an eval run, later a job), if any
        return _Call(messages, purpose, max_tokens, temperature, json_mode,
                     run_id if run_id is not None else current_run_id())

    async def _run(self, call: "_Call", once: Attempt) -> Any:
        """Providers in order, retries, switch-instead-of-wait, last resorts (rules 1-5 at the top)."""
        purpose = call.purpose
        order = self.providers_for(purpose)
        deferred: list[tuple[ChatProvider, float]] = []     # providers that asked for a long wait

        for i, provider in enumerate(order):
            is_last = i == len(order) - 1
            # Switch instead of waiting: if this provider asks for a long wait and another provider
            # is still untried, move on now and come back to this one only if all others fail.
            switch_over = None if is_last else self.wait_for.get(purpose, self.switch_after_s)
            result, wait = await self._try_provider(provider, call, switch_over, once)
            if result is not None:
                return result
            if wait is not None:
                deferred.append((provider, wait))

        for provider, wait in deferred:                     # last resort: wait after all
            log.warning("all other providers failed; waiting %.1fs for %s", wait, provider.name)
            await self._sleep(wait)
            result, _ = await self._try_provider(provider, call, None, once)
            if result is not None:
                return result

        raise AllProvidersFailed(call.errors)

    async def _try_provider(self, provider: ChatProvider, call: "_Call", switch_over: float | None,
                            once: Attempt) -> tuple[Any | None, float | None]:
        """
        Try ONE provider (first try + retries) with `once`. Returns (result, None) on success,
        (None, wait) if it asked for a wait longer than `switch_over` (caller may switch),
        or (None, None) if it failed for good.
        """
        for attempt in range(1, self.max_retries + 2):     # first try + retries
            call.attempts += 1
            start = time.perf_counter()
            try:
                return await once(provider, attempt, start), None
            except ProviderError as error:
                await self._record(call.purpose, provider, attempt, start, call.run_id, error=error)
                call.errors.append(error)
                if error.retryable and attempt <= self.max_retries:
                    delay = self._backoff(attempt, error.retry_after)
                    if switch_over is not None and delay > switch_over:
                        log.warning("%s asks to wait %.1fs (%s): switching to the next provider",
                                    provider.name, delay, type(error).__name__)
                        return None, delay
                    log.warning("%s failed (%s), retry %d in %.1fs", provider.name,
                                type(error).__name__, attempt, delay)
                    await self._sleep(delay)   # time.sleep would freeze the event loop
                    continue
                log.warning("%s gave up (%s), trying next provider", provider.name, type(error).__name__)
                return None, None
        return None, None

    async def _finished(self, call: "_Call", provider: ChatProvider, attempt: int, start: float,
                        response: ProviderResponse) -> LlmResult:
        """A successful answer: record its usage, return it as an LlmResult."""
        latency = time.perf_counter() - start
        billed_output = response.output_tokens + response.thinking_tokens
        cost = provider.cost(response.input_tokens, billed_output)
        await self._record(call.purpose, provider, attempt, start, call.run_id, response=response, cost=cost)
        return LlmResult(
            text=response.text, provider=provider.name, model=provider.model,
            finish_reason=response.finish_reason, input_tokens=response.input_tokens,
            output_tokens=response.output_tokens, thinking_tokens=response.thinking_tokens,
            latency_s=round(latency, 3), cost_usd=cost, attempts=call.attempts,
        )

    async def _record_estimate(self, call: "_Call", provider: ChatProvider, attempt: int, start: float,
                               written: str) -> None:
        """A stream that stopped before the provider's token count arrived: estimate (~4 characters per
        token) and record it as used, so leaving mid-answer is not a way around the daily budget."""
        prompt = sum(len(str(m.get("content", ""))) for m in call.messages)
        estimate = ProviderResponse(text=written, finish_reason="stopped",
                                    input_tokens=prompt // 4, output_tokens=len(written) // 4, thinking_tokens=0)
        await self._record(call.purpose, provider, attempt, start, call.run_id, response=estimate,
                           cost=provider.cost(estimate.input_tokens, estimate.output_tokens),
                           note="stream stopped early; tokens estimated")

    # ---------- validated JSON ----------

    async def complete_structured(self, messages: list[dict], schema: type[T], *, purpose: str,
                                  max_tokens: int = 4000, temperature: float | None = 0,
                                  max_validation_retries: int = 2,
                                  run_id: int | None = None) -> tuple[T, LlmResult]:
        """Return (validated object, LlmResult of the successful call)."""
        conversation = _with_schema_instruction(messages, schema)
        last_error = ""
        for _ in range(max_validation_retries + 1):
            result = await self.complete(conversation, purpose=purpose, max_tokens=max_tokens,
                                         temperature=temperature, json_mode=True, run_id=run_id)
            try:
                return schema.model_validate_json(extract_json(result.text)), result
            except ValidationError as e:
                last_error = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'json'}: {err['msg']}"
                                       for err in e.errors()[:6])
                log.warning("invalid structured output for %s: %s", purpose, last_error)
                conversation = conversation + [
                    {"role": "assistant", "content": result.text},
                    {"role": "user", "content": f"That JSON was invalid: {last_error}. "
                                                "Reply again with corrected JSON only."},
                ]
        raise StructuredOutputError(f"{purpose}: output never matched {schema.__name__}: {last_error}")

    # ---------- helpers ----------

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None and retry_after <= self.backoff_max_s:
            return retry_after
        exponential = self.backoff_base_s * (2 ** (attempt - 1))         # 1s, 2s, 4s ...
        return min(exponential + random.uniform(0, self.backoff_base_s), self.backoff_max_s)

    async def _record(self, purpose, provider, attempt, start, run_id, *, response=None, cost=None,
                      error: ProviderError | None = None, note: str | None = None):
        if self.usage_sink is None:
            return
        record = LlmCallRecord(
            purpose=purpose, provider=provider.name, model=provider.model,
            status="error" if error else "ok", attempt=attempt, run_id=run_id, user_id=current_user_id(),
            latency_ms=int((time.perf_counter() - start) * 1000),
            input_tokens=response.input_tokens if response else 0,
            output_tokens=response.output_tokens if response else 0,
            thinking_tokens=response.thinking_tokens if response else 0,
            cost_usd=cost, error=str(error)[:500] if error else note,
        )
        try:
            await self.usage_sink(record)
        except Exception:   # logging usage must never break the actual LLM call
            log.exception("failed to record LLM usage")


def extract_json(text: str) -> str:
    """Models sometimes wrap JSON in ```json fences or add a sentence; keep the {...} part."""
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


def _with_schema_instruction(messages: list[dict], schema: type[BaseModel]) -> list[dict]:
    """Tell the model the exact fields (a schema alone controls format, not meaning: P9 lesson)."""
    instruction = ("Reply with ONE JSON object only, no other text. It must match this JSON Schema "
                   "(field descriptions explain what each field means):\n"
                   + json.dumps(schema.model_json_schema()))
    messages = [dict(m) for m in messages]
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = f"{messages[0]['content']}\n\n{instruction}"
    else:
        messages.insert(0, {"role": "system", "content": instruction})
    return messages
