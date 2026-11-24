"""Upsert and topic links against the real Postgres (rolled back after each test)."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import get_settings
from app.domain.schemas import PaperRecord
from app.repositories.models import Paper
from app.repositories.paper_repository import PaperRepository
from app.repositories.topic_repository import TopicRepository
from app.repositories.user_repository import UserRepository

pytestmark = pytest.mark.integration


@pytest.fixture
async def session():
    engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
    async with engine.connect() as connection:
        transaction = await connection.begin()
        db = AsyncSession(bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False)
        yield db
        await db.close()
        await transaction.rollback()          # undo everything the test did
    await engine.dispose()


async def test_upsert_creates_then_fills_missing_fields(session):
    repo = PaperRepository(session)
    first = await repo.upsert_from_record(PaperRecord(source="openalex", title="Triceps study", doi="10.77/A",
                                                      citation_count=5, landing_url="https://openalex.org/W1"))
    again = await repo.upsert_from_record(PaperRecord(source="europepmc", title="Triceps study.", doi="10.77/a",
                                                      pmid="770001", abstract="Text", citation_count=9,
                                                      pdf_url="https://e.org/p.pdf"))
    assert again.id == first.id
    assert (again.pmid, again.abstract, again.citation_count) == ("770001", "Text", 9)
    assert again.source_urls == {"openalex": "https://openalex.org/W1", "pdf": "https://e.org/p.pdf"}


async def test_upsert_does_not_steal_an_id_owned_by_another_paper(session):
    repo = PaperRepository(session)
    await repo.add(Paper(title="Other paper", pmid="880001"))
    mine = await repo.upsert_from_record(PaperRecord(source="openalex", title="Mine", doi="10.88/m"))
    updated = await repo.upsert_from_record(PaperRecord(source="s2", title="Mine", doi="10.88/m", pmid="880001"))
    assert updated.id == mine.id and updated.pmid is None      # no unique-constraint crash


async def test_link_paper_merges_found_by(session):
    user = await UserRepository(session).get_or_create("test@example.com")
    topics = TopicRepository(session)
    topic = await topics.create("triceps", user_id=user.id, field="sports science")
    paper = await PaperRepository(session).add(Paper(title="P"))
    await topics.link_paper(topic.id, paper.id, [{"source": "openalex", "query": "q1"}])
    await topics.link_paper(topic.id, paper.id, [{"source": "openalex", "query": "q1"},
                                                 {"source": "europepmc", "query": "q2"}])
    assert [p.id for p in await topics.papers_for_topic(topic.id)] == [paper.id]
