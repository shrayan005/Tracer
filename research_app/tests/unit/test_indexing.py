"""Step 7.6a: passages, Gemini embeddings and indexing. Fakes only: no network, no database, no tokens."""

import json
import math
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest

from app.domain.errors import ProviderRejected
from app.integrations.embeddings.gemini import GeminiEmbedder, MinuteBudget, retry_delay
from app.integrations.llm.run_context import run_scope, user_scope
from app.services import index_service as ix
from app.services.chunking import MAX, TARGET, chunk_paper, sentences, split_text
from app.services.index_service import IndexService, group_papers

S = "Creatine increased leg press strength by 12% in older adults over 12 weeks of training. "


# ---------- passages ----------

def test_passages_follow_sentence_ends_with_one_sentence_overlap():
    text = " ".join(f"Sentence number {i} says something about muscle strength and creatine dosing." for i in range(60))
    parts = split_text(text)
    assert len(parts) > 3 and all(len(p) <= TARGET + 120 for p in parts)
    assert all(p.endswith(".") for p in parts)                        # never cut mid-sentence
    for a, b in zip(parts, parts[1:]):
        assert sentences(a)[-1] == sentences(b)[0]                    # the overlap sentence


def test_a_huge_sentence_is_cut_hard_and_nothing_is_lost():
    blob = "x" * (MAX * 2 + 10)
    assert [len(p) for p in split_text(blob)] == [MAX, MAX, 10]


def test_a_paper_skips_references_and_front_matter_and_keeps_its_abstract():
    sections = [("front", "Authors", "Andrea Brose, McMaster University, Hamilton, Canada. " * 3),
                ("methods", "Methods", S * 3), ("references", "References", "1. Smith J. Creatine. 2001. " * 10),
                ("results", "Results", "Too short.")]
    passages = chunk_paper(sections, abstract="We sought to determine whether creatine monohydrate supplementation "
                                              "enhances strength gains in older adults.")
    assert [p.kind for p in passages] == ["abstract", "methods"]
    assert [p.position for p in passages] == [0, 1]
    assert chunk_paper([], abstract=None) == []                       # nothing to search
    assert chunk_paper([("abstract", "Abstract", S * 2)], abstract=S * 2)[0].kind == "abstract"   # not twice


# ---------- the Gemini embedder ----------

def fake_gemini(script):
    """A fake API: each request takes the next answer from `script` (a status code, or 'ok')."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        status = script.pop(0) if script else "ok"
        if status != "ok":
            return httpx.Response(status, text="nope")
        n = len(json.loads(request.content)["requests"])
        return httpx.Response(200, json={"embeddings": [{"values": [3.0, 4.0] + [0.0] * 6} for _ in range(n)]})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen


async def no_sleep(_s):
    pass


async def test_embedder_batches_normalises_and_sends_the_task_type():
    client, seen = fake_gemini([])
    records = []

    async def sink(r):
        records.append(r)
    embedder = GeminiEmbedder(client, "KEY", dimensions=8, usage_sink=sink, sleep=no_sleep,
                              per_minute=100, budget=MinuteBudget(10_000))      # own budget: no real waiting
    with run_scope(42), user_scope(7):
        vectors = await embedder.embed([f"passage {i}" for i in range(230)], "document")
    assert len(vectors) == 230 and len(seen) == 3                    # batches of 100
    assert math.isclose(math.hypot(*vectors[0]), 1.0)                  # unit length: [0.6, 0.8, 0, ...]
    body = json.loads(seen[0].content)["requests"][0]
    assert body["taskType"] == "RETRIEVAL_DOCUMENT" and body["outputDimensionality"] == 8
    assert seen[0].headers["x-goog-api-key"] == "KEY" and "KEY" not in str(seen[0].url)   # key in a header
    assert {(r.purpose, r.run_id, r.user_id) for r in records} == {("embed", 42, 7)}       # counts to the budget
    await embedder.embed(["what dose?"], "query")
    assert json.loads(seen[-1].content)["requests"][0]["taskType"] == "RETRIEVAL_QUERY"


async def test_the_minute_budget_waits_before_the_free_tier_limit():
    clock = [0.0]
    waits = []

    async def fake_sleep(seconds):                  # waiting moves the fake clock forward
        waits.append(round(seconds, 1))
        clock[0] += seconds
    budget = MinuteBudget(90, clock=lambda: clock[0])
    await budget.take(90, fake_sleep)               # the first 90 go at once
    clock[0] = 10
    await budget.take(40, fake_sleep)               # 90 + 40 > 90 in the same minute: wait until it frees up
    assert waits == [50.5]


def test_googles_wait_time_is_read_from_the_error_body():
    body = {"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "25s"}]}}
    assert retry_delay(httpx.Response(429, json=body)) == 26.0
    assert retry_delay(httpx.Response(429, headers={"retry-after": "7"})) == 7.0
    assert retry_delay(httpx.Response(429, text="no json")) is None


async def test_embedder_retries_rate_limits_but_not_bad_requests():
    client, seen = fake_gemini([429, 503, "ok"])
    vectors = await GeminiEmbedder(client, "KEY", dimensions=8, sleep=no_sleep,
                                   budget=MinuteBudget(10_000)).embed(["a"], "document")
    assert len(vectors) == 1 and len(seen) == 3
    client, seen = fake_gemini([400])
    with pytest.raises(ProviderRejected):
        await GeminiEmbedder(client, "KEY", dimensions=8, sleep=no_sleep,
                             budget=MinuteBudget(10_000)).embed(["a"], "document")
    assert len(seen) == 1                                             # a bad request is not retried


# ---------- the index service ----------

STORE: dict[int, tuple[str, int]] = {}         # paper id -> (version, passages)
TEXTS = {1: S * 30, 2: S * 3, 3: None}         # paper 3 has no text at all


class FakeChunks:
    def __init__(self, session):
        pass

    async def versions(self, paper_ids):
        return {p: STORE[p][0] for p in paper_ids if p in STORE}

    async def replace(self, paper_id, passages, vectors, version):
        assert len(passages) == len(vectors)
        STORE[paper_id] = (version, len(passages))


class FakePapers:
    def __init__(self, session):
        pass

    async def get(self, paper_id):
        return SimpleNamespace(abstract=TEXTS[paper_id])


class NoDocuments:
    def __init__(self, session):
        pass

    async def for_paper(self, paper_id):
        return []


class CountingEmbedder:
    name, dimensions = "fake:8", 8

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    async def embed(self, texts, task):
        self.calls.append(len(texts))
        if self.fail:
            raise ProviderRejected("gemini-embed", "quota")
        return [[0.0] * 8 for _ in texts]


@asynccontextmanager
async def fake_uow():
    yield object()


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    STORE.clear()
    monkeypatch.setattr(ix, "ChunkRepository", FakeChunks)
    monkeypatch.setattr(ix, "PaperRepository", FakePapers)
    monkeypatch.setattr(ix, "DocumentRepository", NoDocuments)


async def test_papers_are_indexed_once_and_embedded_together():
    embedder = CountingEmbedder()
    service = IndexService(fake_uow, embedder)
    first = await service.index([1, 2, 3])
    assert (first.indexed, first.empty, first.reused) == (2, 1, 0)
    assert len(embedder.calls) == 1                                   # both papers in ONE embed call
    assert STORE[1][0] == "fake:8|chunker-1" and first.passages == STORE[1][1] + STORE[2][1]
    again = await service.index([1, 2, 3])
    assert (again.indexed, again.reused) == (0, 2) and len(embedder.calls) == 1   # nothing re-embedded
    await service.index([1], refresh=True)
    assert len(embedder.calls) == 2


async def test_an_embedding_failure_is_reported_not_raised():
    result = await IndexService(fake_uow, CountingEmbedder(fail=True)).index([1, 2])
    assert result.indexed == 0 and len(result.failed) == 2 and STORE == {}


def test_papers_are_grouped_by_passages_and_never_split():
    ps = lambda n: [None] * n                                                     # noqa: E731
    assert group_papers({1: ps(50), 2: ps(30), 3: ps(40), 4: ps(200)}, 90) == [[1, 2], [3], [4]]


async def test_progress_is_saved_group_by_group_before_a_failure():
    class FailsSecondTime(CountingEmbedder):
        async def embed(self, texts, task):
            if self.calls:
                self.calls.append(len(texts))
                raise ProviderRejected("gemini-embed", "quota")
            return await super().embed(texts, task)
    result = await IndexService(fake_uow, FailsSecondTime(), group_size=2).index([1, 2])   # 3 + 1 passages: 2 groups
    assert result.indexed == 1 and list(STORE) == [1]          # paper 1 is kept although paper 2 failed
    assert result.failed == ["paper 2: [gemini-embed] quota"]


async def test_a_long_wait_from_google_fails_fast_instead_of_hanging():
    # measured (7.7): an unbounded wait hung an eval run for 15+ minutes
    def handler(request):
        body = {"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "3600s"}]}}
        return httpx.Response(429, json=body)
    slept = []

    async def record_sleep(s):
        slept.append(s)
    embedder = GeminiEmbedder(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "KEY", dimensions=8,
                              sleep=record_sleep, budget=MinuteBudget(10_000))
    with pytest.raises(Exception, match="asks to wait 3601s"):
        await embedder.embed(["a"], "query")
    assert slept == []                                                # it never slept for an hour
