"""Unit tests for LlmGateway: retries, fallback, usage records, structured output. No network.

The gateway is async, so tests are `async def` and `await` it.
pyproject.toml sets asyncio_mode = "auto", so pytest runs them in an event loop automatically.
"""

import pytest
from pydantic import BaseModel

from app.domain.errors import (AllProvidersFailed, ProviderRateLimited, ProviderRejected,
                               ProviderUnavailable, StreamInterrupted, StructuredOutputError)
from app.integrations.llm.gateway import LlmGateway, extract_json
from tests.fakes import FakeProvider

MSG = [{"role": "user", "content": "hi"}]


async def no_sleep(_seconds):
    """Async stand-in for asyncio.sleep that doesn't wait."""


def make_gateway(*providers, max_retries=2):
    records, sleeps = [], []

    async def fake_sleep(seconds):
        sleeps.append(seconds)          # remember how long it WOULD have waited

    async def fake_sink(record):
        records.append(record)          # instead of saving to the database

    gateway = LlmGateway(list(providers), usage_sink=fake_sink, max_retries=max_retries,
                         backoff_base_s=1.0, sleep=fake_sleep)
    return gateway, records, sleeps


async def test_success_first_try():
    gw, records, sleeps = make_gateway(FakeProvider("a", ["hello"]))
    result = await gw.complete(MSG, purpose="test")
    assert result.text == "hello" and result.provider == "a" and result.attempts == 1
    assert result.thinking_tokens == 2
    assert [r.status for r in records] == ["ok"]
    assert sleeps == []


async def test_retries_temporary_error_on_same_provider():
    a = FakeProvider("a", [ProviderRateLimited("a", "429"), ProviderUnavailable("a", "503"), "hello"])
    gw, records, sleeps = make_gateway(a)
    result = await gw.complete(MSG, purpose="test")
    assert result.text == "hello" and result.attempts == 3 and a.calls == 3
    assert [r.status for r in records] == ["error", "error", "ok"]
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0] - 1   # backoff grows (with jitter)


async def test_respects_retry_after():
    a = FakeProvider("a", [ProviderRateLimited("a", "429", status=429, retry_after=7), "ok"])
    gw, _, sleeps = make_gateway(a)
    await gw.complete(MSG, purpose="test")
    assert sleeps == [7]


async def test_falls_back_after_retries_exhausted():
    a = FakeProvider("a", [ProviderUnavailable("a", "down")] * 3)
    b = FakeProvider("b", ["from b"])
    gw, records, _ = make_gateway(a, b, max_retries=2)
    result = await gw.complete(MSG, purpose="test")
    assert result.provider == "b" and a.calls == 3 and result.attempts == 4
    assert [(r.provider, r.status) for r in records] == [("a", "error")] * 3 + [("b", "ok")]


async def test_permanent_error_skips_provider_without_retry():
    a = FakeProvider("a", [ProviderRejected("a", "model not found", status=404)])
    b = FakeProvider("b", ["from b"])
    gw, _, sleeps = make_gateway(a, b)
    assert (await gw.complete(MSG, purpose="test")).provider == "b"
    assert a.calls == 1 and sleeps == []


async def test_all_providers_fail():
    a = FakeProvider("a", [ProviderRejected("a", "bad key", status=401)])
    b = FakeProvider("b", [ProviderUnavailable("b", "down")] * 3)
    gw, _, _ = make_gateway(a, b)
    with pytest.raises(AllProvidersFailed) as info:
        await gw.complete(MSG, purpose="test")
    assert len(info.value.errors) == 4


async def test_usage_sink_failure_does_not_break_call():
    async def broken_sink(_):
        raise RuntimeError("db down")
    gw = LlmGateway([FakeProvider("a", ["hello"])], usage_sink=broken_sink, sleep=no_sleep)
    assert (await gw.complete(MSG, purpose="test")).text == "hello"


class Person(BaseModel):
    name: str
    age: int


async def test_structured_output_valid():
    a = FakeProvider("a", ['Here you go: ```json {"name": "Ram", "age": 21} ```'])
    gw, _, _ = make_gateway(a)
    person, result = await gw.complete_structured(MSG, Person, purpose="test")
    assert person == Person(name="Ram", age=21)
    assert a.last_kwargs["json_mode"] is True
    assert "JSON Schema" in a.last_kwargs["messages"][0]["content"]


async def test_structured_output_retries_with_feedback():
    a = FakeProvider("a", ['{"name": "Ram"}', '{"name": "Ram", "age": 21}'])
    gw, _, _ = make_gateway(a)
    person, _ = await gw.complete_structured(MSG, Person, purpose="test")
    assert person.age == 21
    feedback = a.last_kwargs["messages"][-1]["content"]
    assert "invalid" in feedback and "age" in feedback      # the model was told what was wrong


async def test_structured_output_gives_up():
    a = FakeProvider("a", ['{"nope": 1}'] * 3)
    gw, _, _ = make_gateway(a)
    with pytest.raises(StructuredOutputError):
        await gw.complete_structured(MSG, Person, purpose="test", max_validation_retries=2)


def test_extract_json():
    # plain function: nothing to await
    assert extract_json('Sure! {"a": 1} hope that helps') == '{"a": 1}'
    assert extract_json("no json here") == "no json here"


# ---------- routes: per-task provider order ----------

async def test_route_tries_the_preferred_provider_first():
    a, b = FakeProvider("a", ["from a"]), FakeProvider("b", ["from b"])
    gw = LlmGateway([a, b], sleep=no_sleep, routes={"summarize": ["b"]})
    assert (await gw.complete(MSG, purpose="summarize")).provider == "b"
    assert a.calls == 0


async def test_purpose_without_route_uses_normal_order():
    a, b = FakeProvider("a", ["from a"]), FakeProvider("b", ["from b"])
    gw = LlmGateway([a, b], sleep=no_sleep, routes={"summarize": ["b"]})
    assert (await gw.complete(MSG, purpose="screen")).provider == "a"


async def test_route_falls_back_to_the_others_and_ignores_unknown_names():
    a = FakeProvider("a", ["from a"])
    b = FakeProvider("b", [ProviderRejected("b", "bad key")])
    gw = LlmGateway([a, b], sleep=no_sleep, routes={"summarize": ["missing", "b"]})
    assert [p.name for p in gw.providers_for("summarize")] == ["b", "a"]
    assert (await gw.complete(MSG, purpose="summarize")).provider == "a"


# ---------- switch instead of waiting ----------

def switching_gateway(*providers, switch_after_s=5.0):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    return LlmGateway(list(providers), max_retries=2, backoff_base_s=1.0, sleep=fake_sleep,
                      switch_after_s=switch_after_s), sleeps


async def test_long_wait_switches_to_next_provider_without_sleeping():
    a = FakeProvider("a", [ProviderRateLimited("a", "slow down", retry_after=18)])
    b = FakeProvider("b", ["from b"])
    gw, sleeps = switching_gateway(a, b)
    assert (await gw.complete(MSG, purpose="screen")).provider == "b"
    assert sleeps == [] and a.calls == 1


async def test_short_wait_still_retries_same_provider():
    a = FakeProvider("a", [ProviderRateLimited("a", "slow down", retry_after=2), "from a"])
    b = FakeProvider("b", ["from b"])
    gw, sleeps = switching_gateway(a, b)
    assert (await gw.complete(MSG, purpose="screen")).provider == "a"
    assert sleeps == [2] and b.calls == 0


async def test_skipped_provider_is_the_last_resort():
    a = FakeProvider("a", [ProviderRateLimited("a", "slow down", retry_after=18), "from a after waiting"])
    b = FakeProvider("b", [ProviderRejected("b", "bad key")])
    gw, sleeps = switching_gateway(a, b)
    result = await gw.complete(MSG, purpose="screen")
    assert result.text == "from a after waiting"
    assert sleeps == [18]                                    # waited only once all others had failed
    assert result.attempts == 3                              # a (429) -> b (rejected) -> a again


async def test_last_provider_waits_instead_of_switching():
    a = FakeProvider("a", [ProviderRateLimited("a", "slow down", retry_after=18), "from a"])
    gw, sleeps = switching_gateway(a)
    assert (await gw.complete(MSG, purpose="screen")).provider == "a"
    assert sleeps == [18]


async def test_wait_for_keeps_a_quality_sensitive_task_on_its_routed_model():
    # measured (6.6): connections made by a fallback model found far fewer links; connect waits for groq
    a = FakeProvider("a", [ProviderRateLimited("a", "slow down", retry_after=18), "from a"])
    b = FakeProvider("b", ["from b"])
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
    gw = LlmGateway([a, b], max_retries=2, sleep=fake_sleep, switch_after_s=5.0, wait_for={"connect": 60.0})
    assert (await gw.complete(MSG, purpose="connect")).provider == "a" and sleeps == [18] and b.calls == 0
    a2 = FakeProvider("a", [ProviderRateLimited("a", "slow down", retry_after=18)])
    gw2 = LlmGateway([a2, FakeProvider("b", ["from b"])], max_retries=2, sleep=fake_sleep, switch_after_s=5.0,
                     wait_for={"connect": 60.0})
    assert (await gw2.complete(MSG, purpose="screen")).provider == "b"        # other tasks still switch fast



async def test_each_call_records_whose_budget_it_counts_against():
    from app.integrations.llm.run_context import run_scope, user_scope
    records = []

    async def sink(record):
        records.append(record)
    gw = LlmGateway([FakeProvider("a", ["one", "two"])], usage_sink=sink)
    with run_scope(42), user_scope(7):
        await gw.complete(MSG, purpose="screen")
    await gw.complete(MSG, purpose="screen")                        # outside any user: an admin/eval call
    assert [(r.run_id, r.user_id) for r in records] == [(42, 7), (None, None)]


# ---------- streaming (SSE step 3) ----------

async def collect(stream) -> tuple[list[str], object]:
    """Pieces and the final LlmResult of gateway.stream()."""
    pieces, result = [], None
    async for item in stream:
        if isinstance(item, str):
            pieces.append(item)
        else:
            result = item
    return pieces, result


async def test_stream_sends_pieces_then_the_result_with_usage():
    gw, records, _ = make_gateway(FakeProvider("a", ["Creatine raised lean mass"]))
    pieces, result = await collect(gw.stream(MSG, purpose="chat"))
    assert pieces == ["Creatine", " raised", " lean", " mass"]
    assert result.text == "Creatine raised lean mass" and result.provider == "a" and result.thinking_tokens == 2
    assert [r.status for r in records] == ["ok"] and records[0].output_tokens == 5


async def test_stream_retries_and_falls_back_before_the_first_piece():
    a = FakeProvider("a", [ProviderRateLimited("a", "429"), ProviderRejected("a", "bad key", 401)])
    b = FakeProvider("b", ["from b"])
    gw, records, sleeps = make_gateway(a, b)
    pieces, result = await collect(gw.stream(MSG, purpose="chat"))
    assert "".join(pieces) == "from b" and result.provider == "b" and result.attempts == 3
    assert [r.status for r in records] == ["error", "error", "ok"] and sleeps == [pytest.approx(1.0, abs=1.0)]


async def test_a_stream_that_breaks_mid_way_is_an_error_not_a_silent_switch():
    a = FakeProvider("a", [("Creatine raised", ProviderUnavailable("a", "connection reset"))])
    b = FakeProvider("b", ["never used"])
    gw, records, _ = make_gateway(a, b)
    pieces = []
    with pytest.raises(StreamInterrupted, match="stopped mid-way"):
        async for item in gw.stream(MSG, purpose="chat"):
            pieces.append(item)
    assert pieces == ["Creatine", " raised"]          # the user already saw these: no other model takes over
    assert b.calls == 0 and [r.status for r in records] == ["error"]


async def test_a_reader_that_stops_early_still_pays_estimated_tokens():
    gw, records, _ = make_gateway(FakeProvider("a", ["one two three four"]))
    stream = gw.stream([{"role": "user", "content": "x" * 400}], purpose="chat")
    async for _piece in stream:
        break                                           # e.g. the user closed the tab
    await stream.aclose()
    assert [r.status for r in records] == ["ok"]        # "ok" rows are what the daily budget counts
    assert records[0].input_tokens == 100 and "estimated" in records[0].error


# ---------- traces: prompt + answer text for chosen purposes ----------

async def test_traced_purposes_keep_prompt_and_answer_others_do_not():
    records = []

    async def sink(record):
        records.append(record)

    gw = LlmGateway([FakeProvider("a", ["done", "plain"])], usage_sink=sink, sleep=no_sleep,
                    trace_purposes={"search_review"})
    await gw.complete([{"role": "system", "content": "review"}, {"role": "user", "content": "titles"}],
                      purpose="search_review")
    await gw.complete([{"role": "user", "content": "my private question"}], purpose="chat")
    assert records[0].prompt == "### system\nreview\n\n### user\ntitles" and records[0].response == "done"
    assert records[1].prompt is None and records[1].response is None


async def test_long_traces_are_cut_in_the_middle():
    records = []

    async def sink(record):
        records.append(record)

    gw = LlmGateway([FakeProvider("a", ["x" * 50])], usage_sink=sink, sleep=no_sleep,
                    trace_purposes={"plan"}, trace_max_chars=20)
    await gw.complete([{"role": "user", "content": "hi"}], purpose="plan")
    assert records[0].response.startswith("x" * 10) and records[0].response.endswith("x" * 10)
    assert "30 characters cut" in records[0].response
