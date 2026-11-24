"""Extraction service against the real Postgres (rolled back) and a temporary file folder."""

from contextlib import asynccontextmanager

import pymupdf
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import get_settings
from app.domain.errors import NotFoundError
from app.integrations.storage.file_storage import LocalFileStorage, make_key, sha256_of
from app.repositories.document_repository import DocumentRepository
from app.repositories.models import Document, Paper
from app.repositories.paper_repository import PaperRepository
from app.services.extraction_service import ExtractionService

pytestmark = pytest.mark.integration

JATS = b"""<?xml version="1.0"?><article><front><article-meta><abstract><p>Abstract text.</p></abstract>
</article-meta></front><body>
<sec><title>Introduction</title><p>Why.</p></sec>
<sec><title>Methods</title><p>Ten participants did elbow extensions.</p></sec>
<sec><title>Results</title><p>Force rose by 12%.</p></sec></body></article>"""


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


def service_on(session, tmp_path) -> ExtractionService:
    @asynccontextmanager
    async def unit_of_work():
        yield session
        await session.flush()
    return ExtractionService(LocalFileStorage(tmp_path), unit_of_work)


async def stored_document(session, tmp_path, data: bytes, kind: str, paper_id: int | None) -> Document:
    sha = sha256_of(data)
    key = make_key(sha, kind)
    await LocalFileStorage(tmp_path).save(key, data)
    return await DocumentRepository(session).add(Document(
        sha256=sha, storage_key=key, content_type=f"application/{kind}", size_bytes=len(data),
        origin="open_access", paper_id=paper_id))


async def test_xml_is_extracted_into_sections(session, tmp_path):
    paper = await PaperRepository(session).add(Paper(title="Triceps"))
    document = await stored_document(session, tmp_path, JATS, "xml", paper.id)
    result = await service_on(session, tmp_path).extract(document.id)

    assert result.status == "extracted" and not result.reused
    assert result.kinds == ["abstract", "introduction", "methods", "results"]
    sections = await DocumentRepository(session).sections(document.id)
    assert [s.position for s in sections] == [0, 1, 2, 3]
    assert sections[2].kind == "methods" and "Ten participants" in sections[2].text
    assert document.status == "extracted" and paper.fulltext_status == "full"


async def test_second_run_reuses_and_force_redoes(session, tmp_path):
    document = await stored_document(session, tmp_path, JATS, "xml", None)
    service = service_on(session, tmp_path)
    first = await service.extract(document.id)
    again = await service.extract(document.id)
    assert again.reused and again.sections == first.sections
    redone = await service.extract(document.id, force=True)
    assert not redone.reused and redone.sections == first.sections
    assert len(await DocumentRepository(session).sections(document.id)) == first.sections   # replaced, not doubled


async def test_scanned_pdf_is_marked_failed_with_the_reason(session, tmp_path):
    doc = pymupdf.open()
    doc.new_page().draw_rect(pymupdf.Rect(50, 50, 500, 700), fill=(0.9, 0.9, 0.9))     # no text at all
    document = await stored_document(session, tmp_path, doc.tobytes(), "pdf", None)
    result = await service_on(session, tmp_path).extract(document.id)
    assert result.status == "failed" and "scanned PDF" in result.error
    assert document.status == "failed" and "scanned PDF" in document.error


async def test_unknown_document(session, tmp_path):
    with pytest.raises(NotFoundError):
        await service_on(session, tmp_path).extract(999_999_999)
