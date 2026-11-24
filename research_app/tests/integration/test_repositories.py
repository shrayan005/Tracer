"""
Repository tests against the real Postgres in WSL (async).
Each test runs inside a transaction that is rolled back, so the database stays clean.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import get_settings
from app.domain.schemas import LlmCallRecord
from app.repositories.llm_call_repository import LlmCallRepository
from app.repositories.models import Paper
from app.repositories.paper_repository import PaperRepository
from app.repositories.topic_repository import TopicRepository
from app.repositories.user_repository import UserRepository

pytestmark = pytest.mark.integration


@pytest.fixture
async def session():
    # A fresh engine per test with NullPool: async connections belong to ONE event loop,
    # and pytest gives every test its own loop, so pooled connections must not be shared.
    engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
    async with engine.connect() as connection:
        transaction = await connection.begin()
        db = AsyncSession(bind=connection, join_transaction_mode="create_savepoint",
                          expire_on_commit=False)
        yield db
        await db.close()
        await transaction.rollback()          # undo everything the test did
    await engine.dispose()


async def test_topic_create_and_get(session):
    user = await UserRepository(session).get_or_create("test@example.com")
    repo = TopicRepository(session)
    topic = await repo.create("  biomechanics of triceps  ", user_id=user.id)
    assert topic.id is not None
    assert (await repo.get(topic.id)).query == "biomechanics of triceps"


async def test_paper_lookup_by_any_id_and_doi_case(session):
    repo = PaperRepository(session)
    await repo.add(Paper(title="Triceps EMG study", doi="10.1000/ABC.123", pmid="999001", authors=["Ram, S."]))
    assert (await repo.get_by_doi("10.1000/abc.123")).title == "Triceps EMG study"
    assert (await repo.find_by_any_id(pmid="999001", doi=None)).title == "Triceps EMG study"
    assert await repo.find_by_any_id(arxiv_id="nope") is None


async def test_llm_call_logging_and_cost(session):
    repo = LlmCallRepository(session)
    for cost in (0.001, 0.0025, None):
        await repo.add(LlmCallRecord(purpose="test", provider="groq", model="m", status="ok", attempt=1,
                                     cost_usd=cost, run_id=424242))
    assert await repo.total_cost(run_id=424242) == pytest.approx(0.0035)


async def test_provider_usage_today_counts_failures_and_leaves_embeddings_out_of_tokens(session):
    from datetime import datetime, timedelta, timezone
    user = await UserRepository(session).get_or_create("usage-test@example.com")
    repo = LlmCallRepository(session)

    def call(provider, status="ok", purpose="chat", tokens=0, error=None):
        return LlmCallRecord(purpose=purpose, provider=provider, model="m", status=status, attempt=1,
                             input_tokens=tokens, user_id=user.id, error=error)
    await repo.add(call("gemini", tokens=1000))
    await repo.add(call("gemini", purpose="embed", tokens=5000))          # counted as a call, not as tokens
    await repo.add(call("gemini", tokens=10, error="note: thinking trimmed"))  # an ok call with a note: no error
    await repo.add(call("groq", status="error", error="429 too many requests (first)"))
    await repo.add(call("groq", status="error", error="429 too many requests (newest)"))

    since = datetime.now(timezone.utc) - timedelta(hours=1)
    rows = {r["provider"]: r for r in await repo.user_providers_since(user.id, since)}
    assert (rows["gemini"]["ok_calls"], rows["gemini"]["failed_calls"], rows["gemini"]["tokens"]) == (3, 0, 1010)
    assert rows["gemini"]["last_error"] is None
    assert (rows["groq"]["ok_calls"], rows["groq"]["failed_calls"]) == (0, 2)
    assert rows["groq"]["last_error"] == "429 too many requests (newest)"
    assert await repo.user_providers_since(user.id, datetime.now(timezone.utc) + timedelta(hours=1)) == []
