"""paper_detail (step 5.7) against the real Postgres (rolled back): the right paper, its connections,
its document and sections, and the not-found cases."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import get_settings
from app.domain.errors import NotFoundError
from app.repositories.document_repository import DocumentRepository
from app.repositories.models import Document
from app.repositories.user_repository import UserRepository
from app.services.research_service import ResearchService
from tests.integration.test_synthesis import FakeAnalyzer, inputs, make_service, make_topic, uow_for

pytestmark = pytest.mark.integration


@pytest.fixture
async def session():
    engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
    async with engine.connect() as connection:
        transaction = await connection.begin()
        db = AsyncSession(bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False)
        yield db
        await db.close()
        await transaction.rollback()
    await engine.dispose()


async def test_paper_detail_has_the_paper_its_connections_and_its_document(session):
    topic, papers = await make_topic(session)
    analyzer = FakeAnalyzer({papers[0].id: 25, papers[1].id: 7})
    await make_service(session, analyzer).run(topic.id, "brief", "sport_biomechanics", inputs(papers))
    docs = DocumentRepository(session)
    failed = await docs.add(Document(sha256="a" * 64, storage_key="k1", content_type="application/pdf",
                                     size_bytes=10, origin="open_access", paper_id=papers[1].id, status="failed",
                                     error="no text layer (scanned PDF)"))
    good = await docs.add(Document(sha256="b" * 64, storage_key="k2", content_type="application/xml",
                                   size_bytes=2048, origin="open_access", paper_id=papers[1].id, status="extracted"))
    await docs.replace_sections(good.id, [("methods", "Methods", "m" * 500), ("results", "Results", "r" * 300)])

    detail = await ResearchService(None, None, None, uow_for(session)).paper_detail(topic.id, 2, topic.user_id)
    assert detail.paper.paper_id == papers[1].id and detail.n_papers == 2
    assert len(detail.connections) == 1                                    # the fake connector links both papers
    assert detail.document.status == "extracted" and detail.document.content_type == "application/xml"
    assert [(s.kind, s.characters) for s in detail.document.sections] == [("methods", 500), ("results", 300)]
    assert detail.labels[papers[0].id] and detail.ranks[papers[0].id] == 1
    assert failed.id != good.id


async def test_failed_document_is_shown_when_it_is_the_only_one(session):
    topic, papers = await make_topic(session)
    await DocumentRepository(session).add(Document(
        sha256="c" * 64, storage_key="k3", content_type="application/pdf", size_bytes=10, origin="open_access",
        paper_id=papers[0].id, status="failed", error="no text layer (scanned PDF)"))
    detail = await ResearchService(None, None, None, uow_for(session)).paper_detail(topic.id, 1, topic.user_id)
    assert detail.document.status == "failed" and "scanned" in detail.document.error
    assert detail.connections == []                                         # no synthesis ran for this topic


async def test_unknown_rank_and_other_users_topic(session):
    topic, _ = await make_topic(session)
    service = ResearchService(None, None, None, uow_for(session))
    with pytest.raises(NotFoundError, match="has no paper ranked 99"):
        await service.paper_detail(topic.id, 99, topic.user_id)
    stranger = await UserRepository(session).get_or_create("stranger@example.com")
    with pytest.raises(NotFoundError, match="does not exist"):
        await service.paper_detail(topic.id, 1, stranger.id)
