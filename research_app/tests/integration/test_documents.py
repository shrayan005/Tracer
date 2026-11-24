"""Users, stored documents and who may see them, against the real Postgres (rolled back)."""

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import get_settings
from app.integrations.storage.file_storage import make_key, sha256_of
from app.repositories.document_repository import DocumentRepository
from app.repositories.models import Document, Paper, User, UserDocument
from app.repositories.paper_repository import PaperRepository
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


def document(content: bytes, origin: str, paper_id: int | None = None) -> Document:
    sha = sha256_of(content)
    return Document(sha256=sha, storage_key=make_key(sha, "pdf"), content_type="application/pdf",
                    size_bytes=len(content), origin=origin, paper_id=paper_id)


async def test_get_or_create_user_is_case_insensitive(session):
    users = UserRepository(session)
    first = await users.get_or_create("Ana@Example.COM", display_name="Ana")
    again = await users.get_or_create("  ana@example.com ")
    assert again.id == first.id and first.email == "ana@example.com"


async def test_documents_by_fingerprint_and_by_paper(session):
    paper = await PaperRepository(session).add(Paper(title="Triceps study"))
    docs = DocumentRepository(session)
    pdf = await docs.add(document(b"%PDF-1 test pdf", "open_access", paper.id))
    xml = await docs.add(document(b"<article>test</article>", "open_access", paper.id))
    assert (await docs.get_by_sha256(sha256_of(b"%PDF-1 test pdf"))).id == pdf.id
    assert [d.id for d in await docs.for_paper(paper.id)] == [pdf.id, xml.id]
    assert pdf.status == "stored"                                     # default


async def test_visibility_open_access_vs_private_upload(session):
    users, docs = UserRepository(session), DocumentRepository(session)
    owner = await users.get_or_create("owner@example.com")
    other = await users.get_or_create("other@example.com")
    public = await docs.add(document(b"%PDF-1 public", "open_access"))
    upload = await docs.add(document(b"%PDF-1 private", "upload"))
    await docs.link_user(owner.id, upload.id, original_filename="my paper.pdf")
    await docs.link_user(owner.id, upload.id, original_filename="my paper.pdf")   # twice: still one link

    assert await docs.visible_to(other.id, public.id)                 # open access: everyone
    assert await docs.visible_to(owner.id, upload.id)                 # upload: its uploader...
    assert not await docs.visible_to(other.id, upload.id)             # ...and nobody else
    assert not await docs.visible_to(owner.id, 999_999_999)           # missing document
    links = (await session.scalars(select(UserDocument).where(UserDocument.document_id == upload.id))).all()
    assert len(links) == 1


async def test_deleting_a_user_keeps_the_shared_file(session):
    users, docs = UserRepository(session), DocumentRepository(session)
    owner = await users.get_or_create("leaver@example.com")
    upload = await docs.add(document(b"%PDF-1 shared bytes", "upload"))
    await docs.link_user(owner.id, upload.id)
    await session.execute(delete(User).where(User.id == owner.id))   # ON DELETE CASCADE on the link
    session.expunge_all()
    assert await session.scalar(select(UserDocument).where(UserDocument.document_id == upload.id)) is None
    assert await docs.get(upload.id) is not None                      # the file row itself remains
