"""Step 7.6b: merging searches, checking answers, and the question API. Fakes only: no LLM, no database."""

import json
from datetime import datetime, timezone

import httpx
import pytest

from app.dependencies import get_ask_service, get_job_service, require_user_id
from app.domain.errors import AllProvidersFailed, NotFoundError, NotReadyError
from app.domain.schemas import (Answer, AnswerDraft, AnswerPart, AnswerSentence, AskResult, ChatInfo,
                                ChatMessageInfo, Citation, Readiness, SearchHit)
from app.main import app
from app.services.ask_service import check, plain_text, unescape
from app.services.retrieval_service import Hit, diversify, fuse, or_query

NOW = datetime(2026, 11, 23, tzinfo=timezone.utc)
PASSAGE = "CrM supplementation (5 g/d) resulted in significantly greater increases in fat-free mass (p < .05)."


def hit(cid=1, paper=10, text=PASSAGE, kind="results", score=1.0) -> Hit:
    return Hit(chunk_id=cid, paper_id=paper, kind=kind, title=None, text=text, score=score)


# ---------- retrieval helpers ----------

def test_fusion_rewards_passages_good_in_either_search_and_most_in_both():
    ranked = [cid for cid, _ in fuse(meaning=[1, 2, 3], keyword=[3, 4])]
    assert ranked[0] == 3 and set(ranked) == {1, 2, 3, 4}


def test_keyword_query_matches_any_meaningful_word():
    assert or_query("What daily dose of creatine?") == "creatine | daily | dose | what"
    assert or_query("of the") is None


def test_at_most_three_passages_per_paper():
    hits = [hit(cid=i, paper=1, score=10 - i) for i in range(5)] + [hit(cid=9, paper=2, score=0.1)]
    assert [h.paper_id for h in diversify(hits, k=8, per_paper=3)] == [1, 1, 1, 2]


# ---------- checking answers ----------

def draft(*sentences, answered=True) -> AnswerDraft:
    return AnswerDraft(sentences=[AnswerSentence(text=t, excerpt=e, quote=q) for t, e, q in sentences], answered=answered)


PAPERS = {10: (4, "(Brose et al., 2003)")}


def test_a_supported_sentence_gets_its_citation_and_no_flags():
    answer = check(draft(("Brose et al. found greater fat-free mass gains with 5 g/d.", "E1",
                          "resulted in significantly greater increases in fat-free mass")), {"E1": hit()}, PAPERS)
    part = answer.parts[0]
    assert part.flags == [] and part.citation.label == "(Brose et al., 2003)" and part.citation.rank == 4


def test_invented_quotes_numbers_and_excerpts_are_flagged_not_hidden():
    answer = check(draft(("Gains were 40% larger.", "E1", "gains were forty percent larger in every group"),
                         ("Something else.", "E7", "x")), {"E1": hit()}, PAPERS)
    assert answer.parts[0].flags == ["quote not found in the passage", "number not in the passage: 40"]
    assert answer.parts[1].citation is None and "unknown excerpt (E7)" in answer.parts[1].flags[0]


def test_escaped_unicode_from_the_model_is_repaired_before_checking():
    assert unescape("\\u22655 g/day") == "≥5 g/day"
    passage = "Doses of ≥5 g/day increased lean mass."
    answer = check(draft(("Doses \\u22655 g/day increased lean mass.", "E1", "Doses of \\u22655 g/day")),
                   {"E1": hit(text=passage)}, PAPERS)
    assert answer.parts[0].text.startswith("Doses ≥5") and answer.parts[0].flags == []


def test_no_sentences_means_not_answered():
    answer = check(draft(answered=True), {"E1": hit()}, PAPERS)
    assert answer.answered is False
    assert plain_text(Answer(answered=False, missing="the capital of France")) == \
        "(Not covered: the capital of France)"


# ---------- the API ----------

def message(mid, role, content, answer=None) -> ChatMessageInfo:
    return ChatMessageInfo(id=mid, role=role, content=content, answer=answer, created_at=NOW)


class FakeAsk:
    async def ask(self, topic_id, user_id, question, chat_id=None):
        if topic_id == 5:
            raise NotReadyError("this topic is not prepared for questions yet")
        part = AnswerPart(text="Yes.", citation=Citation(paper_id=10, rank=4, label="(Brose et al., 2003)",
                                                          section="results", quote="greater", chunk_id=1))
        return AskResult(chat_id=chat_id or 1, question=message(1, "user", question),
                         answer=message(2, "assistant", "Yes.", Answer(parts=[part], answered=True)))

    async def start_ask(self, topic_id, user_id, question, chat_id=None):
        result = await self.ask(topic_id, user_id, question, chat_id)       # 409 for topic 5, before any event

        async def steps():
            yield "stage", {"step": "searching", "papers": 10}
            yield "stage", {"step": "reading", "passages": 8, "papers": 4}
            if question.startswith("break"):
                raise AllProvidersFailed(["groq: 503"])
            if question.startswith("crash"):
                raise RuntimeError("secret internal detail")
            yield "stage", {"step": "writing"}
            yield "stage", {"step": "checking"}
            yield "done", result
        return steps()

    async def readiness(self, topic_id, user_id):
        if topic_id != 119:
            raise NotFoundError(f"topic {topic_id} does not exist")
        return Readiness(papers=10, prepared=10, passages=356)

    async def chats(self, user_id, topic_id=None):
        return [ChatInfo(id=1, topic_id=119, title="q", created_at=NOW, updated_at=NOW)]

    async def chat(self, chat_id, user_id):
        raise NotFoundError(f"chat {chat_id} does not exist")

    async def search(self, user_id, query, topic_id=None, k=20):
        return [SearchHit(paper_id=10, topic_id=119, rank=4, label="(Brose et al., 2003)", title="T",
                          section="results", snippet="\u0002HMB\u0003 was given")]


class FakeJobs:
    def __init__(self):
        self.submitted = []

    async def check_budget(self, user_id):
        pass

    async def submit(self, kind, params, user_id, retry_of=None):
        from app.domain.schemas import JobInfo
        self.submitted.append((kind, params))
        return JobInfo(id=3, kind=kind, params=params, status="queued", created_at=NOW)


@pytest.fixture
async def client():
    jobs = FakeJobs()
    app.dependency_overrides.update({get_ask_service: lambda: FakeAsk(), get_job_service: lambda: jobs,
                                     require_user_id: lambda: 1})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        c.jobs = jobs
        yield c
    app.dependency_overrides.clear()


async def test_ask_and_its_errors(client):
    r = await client.post("/api/topics/119/ask", json={"question": "Did creatine help?"})
    assert r.status_code == 200 and r.json()["answer"]["answer"]["parts"][0]["citation"]["rank"] == 4
    r = await client.post("/api/topics/5/ask", json={"question": "Did creatine help?"})
    assert r.status_code == 409 and "not prepared" in r.json()["detail"]
    assert (await client.post("/api/topics/119/ask", json={"question": "a"})).status_code == 422


def events(text: str) -> list[tuple[str, dict]]:
    """Read a Server-Sent Events body back into (event, data) pairs."""
    out = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((fields["event"], json.loads(fields["data"])))
    return out


async def test_the_answer_streams_its_real_steps_then_the_result(client):
    r = await client.post("/api/topics/119/ask/stream", json={"question": "Did creatine help?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    got = events(r.text)
    assert [d.get("step", e) for e, d in got] == ["searching", "reading", "writing", "checking", "done"]
    assert got[-1][1]["answer"]["answer"]["parts"][0]["citation"]["rank"] == 4


async def test_stream_errors_before_and_during_the_stream(client):
    r = await client.post("/api/topics/5/ask/stream", json={"question": "Did creatine help?"})
    assert r.status_code == 409 and "not prepared" in r.json()["detail"]      # before: a normal JSON error
    r = await client.post("/api/topics/119/ask/stream", json={"question": "break it"})
    assert r.status_code == 200 and events(r.text)[-1][0] == "error"           # during: an error event
    detail = events(r.text)[-1][1]["detail"]
    assert "try again" in detail and "groq" not in detail                     # no raw provider messages
    r = await client.post("/api/topics/119/ask/stream", json={"question": "crash it"})
    assert "secret" not in r.text and events(r.text)[-1][0] == "error"         # unknown errors: no internals


async def test_prepare_checks_the_topic_before_queueing(client):
    r = await client.post("/api/topics/119/prepare")
    assert r.status_code == 202 and client.jobs.submitted == [("index", {"topic_id": 119})]
    assert (await client.post("/api/topics/1/prepare")).status_code == 404
    assert len(client.jobs.submitted) == 1


async def test_chats_readiness_and_search(client):
    assert (await client.get("/api/topics/119/readiness")).json()["passages"] == 356
    assert (await client.get("/api/chats?topic_id=119")).json()[0]["id"] == 1
    assert (await client.get("/api/chats/99")).status_code == 404
    assert (await client.get("/api/search?q=HMB")).json()[0]["snippet"] == "\u0002HMB\u0003 was given"


def test_a_repeated_point_from_the_same_quote_is_dropped():
    q = "resulted in significantly greater increases in fat-free mass"
    answer = check(draft(("Creatine increased fat-free mass.", "E1", q),
                         ("Furthermore, fat-free mass rose more with creatine.", "E1", q.upper())), {"E1": hit()}, PAPERS)
    assert [p.text for p in answer.parts] == ["Creatine increased fat-free mass."]


async def test_retrieval_falls_back_to_keywords_when_embedding_fails(monkeypatch):
    from contextlib import asynccontextmanager

    from app.domain.errors import ProviderRateLimited
    from app.services import retrieval_service as rs

    class BrokenEmbedder:
        name, dimensions = "x:8", 8

        async def embed(self, texts, task):
            raise ProviderRateLimited("gemini-embed", "quota")

    class FakeSession:
        def __init__(self):
            self.scalar_calls = 0

        async def execute(self, query, *args):
            if "iterative_scan" in str(query):
                return None
            return [(7, 10, "results", None, PASSAGE)]                 # the rows of the found ids

        async def scalars(self, query):
            self.scalar_calls += 1
            return [7]                                               # only the keyword search runs

    session = FakeSession()

    @asynccontextmanager
    async def uow():
        yield session
    notes = []
    hits = await rs.RetrievalService(uow, BrokenEmbedder()).search("creatine dose", [10], notes=notes)
    assert [h.chunk_id for h in hits] == [7] and hits[0].keyword_rank == 1 and hits[0].meaning_rank is None
    assert session.scalar_calls == 1 and "keyword search only" in notes[0]


def test_a_named_author_focuses_the_search_on_that_paper():
    from app.services.ask_service import focus
    papers = {1: (1, "(Brose et al., 2003)"), 2: (2, "(Stares & Bains, 2019)"), 3: (3, "(Yao et al., 2026)")}
    assert focus("What dose did Brose et al. give?", papers) == [1]
    assert focus("what did stares find?", papers) == [2]
    assert focus("What doses were used?", papers) == [1, 2, 3]            # no author named: all papers
