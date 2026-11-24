"""The HTTP API (step 7.2) with fake services: no LLM, no Redis, no database."""

from datetime import datetime, timezone

import httpx
import pytest

from app.dependencies import get_job_service, get_research_service, require_user_id
from app.domain.errors import ConflictError, NotFoundError, NotReadyError, QuotaExceededError
from app.domain.schemas import JobInfo, TopicSummary, UsageInfo
from app.main import app
from tests.unit.test_report_markdown import make_report as report

NOW = datetime(2026, 11, 23, 10, 0, tzinfo=timezone.utc)
ME, OTHER = 1, 2


def job(job_id=7, kind="research", status="queued", **kw) -> JobInfo:
    return JobInfo(id=job_id, kind=kind, params=kw.pop("params", {"topic": "creatine"}), status=status,
                   created_at=NOW, **kw)


class FakeJobs:
    def __init__(self):
        self.submitted = []

    async def submit(self, kind, params, user_id, retry_of=None):
        if params.get("topic") == "over budget":
            raise QuotaExceededError("your daily LLM budget (250,000 tokens) is used up", retry_after=3600)
        if kind == "research" and any(k == "research" for k, _, u in self.submitted if u == user_id):
            raise ConflictError("you already have a research running: wait for it to finish")
        self.submitted.append((kind, params, user_id))
        return job(len(self.submitted), kind, params=params)

    async def get(self, job_id, user_id):
        if job_id != 7 or user_id != ME:
            raise NotFoundError(f"job {job_id} does not exist")
        return job(7, status="running", progress=[{"at": "2026-11-23T10:00:05", "message": "screening..."}])

    async def list(self, user_id, limit=20):
        return [job(7)] if user_id == ME else []

    async def usage(self, user_id):
        return UsageInfo(tokens_today=40_000, cost_today_usd=0.01, daily_budget=250_000, remaining=210_000,
                         resets_at=NOW)

    async def retry(self, job_id, user_id):
        if job_id == 8:
            raise ConflictError("job 8 is done: only a failed job can be retried")
        return job(9, retry_of=job_id)


class FakeResearch:
    async def load(self, topic_id, user_id=None):
        if topic_id == 5:
            raise NotReadyError("topic 5 has no research results yet (still running, or it failed)")
        if topic_id != 119 or user_id != ME:
            raise NotFoundError(f"topic {topic_id} does not exist")
        return report()

    async def topics(self, user_id, limit=50):
        return [TopicSummary(id=119, query="biomechanics of triceps", field="sports science", created_at=NOW,
                             has_results=True, papers=10, analysed=True)]

    async def paper_detail(self, topic_id, rank, user_id=None):
        raise NotFoundError(f"topic {topic_id} has no paper ranked {rank}")


@pytest.fixture
async def client():
    jobs, research = FakeJobs(), FakeResearch()
    app.dependency_overrides.update({get_job_service: lambda: jobs, get_research_service: lambda: research,
                                     require_user_id: lambda: ME})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        c.jobs = jobs
        yield c
    app.dependency_overrides.clear()


# ---------- starting work ----------

async def test_research_returns_a_queued_job_at_once(client):
    r = await client.post("/api/research", json={"topic": "  creatine in older adults  "})
    assert r.status_code == 202 and r.json()["status"] == "queued"
    assert client.jobs.submitted == [("research", {"topic": "creatine in older adults", "papers": 10}, ME)]


async def test_research_takes_a_paper_count_of_5_10_15_or_20_only(client):
    r = await client.post("/api/research", json={"topic": "creatine in older adults", "papers": 20})
    assert r.status_code == 202 and client.jobs.submitted[-1][1]["papers"] == 20
    r = await client.post("/api/research", json={"topic": "creatine in older adults", "papers": 50})
    assert r.status_code == 422 and len(client.jobs.submitted) == 1      # refused before any job exists


async def test_a_second_research_while_one_runs_is_refused(client):
    await client.post("/api/research", json={"topic": "creatine"})
    r = await client.post("/api/research", json={"topic": "creatine again"})
    assert r.status_code == 409 and "already have a research running" in r.json()["detail"]


async def test_topic_length_is_checked(client):
    assert (await client.post("/api/research", json={"topic": "ab"})).status_code == 422
    assert (await client.post("/api/research", json={"topic": "x" * 301})).status_code == 422


async def test_analyze_checks_the_topic_before_queueing(client):
    r = await client.post("/api/topics/119/analyze", json={"refresh": True})
    assert r.status_code == 202 and client.jobs.submitted == [("analyze", {"topic_id": 119, "refresh": True}, ME)]
    assert (await client.post("/api/topics/404/analyze")).status_code == 404       # not yours / missing
    assert (await client.post("/api/topics/5/analyze")).status_code == 409         # no results yet
    assert len(client.jobs.submitted) == 1                                         # neither was queued


# ---------- following jobs ----------

async def test_job_status_and_progress(client):
    body = (await client.get("/api/jobs/7")).json()
    assert body["status"] == "running" and body["progress"][0]["message"] == "screening..."


async def test_another_users_job_is_a_plain_404(client):
    r = await client.get("/api/jobs/12")
    assert r.status_code == 404 and r.json() == {"detail": "job 12 does not exist"}


async def test_retry(client):
    r = await client.post("/api/jobs/3/retry")
    assert r.status_code == 202 and r.json()["retry_of"] == 3
    assert (await client.post("/api/jobs/8/retry")).status_code == 409


# ---------- reading results ----------

async def test_topic_list_and_report(client):
    topics = (await client.get("/api/topics")).json()
    assert topics[0]["id"] == 119 and topics[0]["papers"] == 10
    r = await client.get("/api/topics/119")
    assert r.status_code == 200 and r.json()["topic_id"] == report().topic_id
    assert (await client.get("/api/topics/1")).status_code == 404
    assert (await client.get("/api/topics/5")).status_code == 409


async def test_report_markdown_is_the_same_text_the_cli_writes(client):
    from app.presentation.report_markdown import render
    r = await client.get("/api/topics/119/report.md")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/markdown")
    assert r.text == render(report()) and 'filename="' in r.headers["content-disposition"]


async def test_unknown_paper_rank_is_404(client):
    assert (await client.get("/api/topics/119/papers/99")).status_code == 404


async def test_the_old_blocking_search_endpoint_is_gone(client):
    assert (await client.post("/api/discovery/search", json={"topic": "x y z"})).status_code == 404



async def test_usage_and_the_budget_over_http(client):
    assert (await client.get("/api/usage")).json()["remaining"] == 210_000
    r = await client.post("/api/research", json={"topic": "over budget"})
    assert r.status_code == 429 and r.headers["retry-after"] == "3600" and "budget" in r.json()["detail"]
