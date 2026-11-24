"""
The fetcher end to end against the real Postgres (rolled back) and a temporary folder,
with a fake finder and a fake downloader (no network).
"""

from contextlib import asynccontextmanager

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import get_settings
from app.domain.errors import DownloadError
from app.domain.schemas import FullTextLink, PaperRecord
from app.integrations.downloader import Downloaded
from app.integrations.storage.file_storage import LocalFileStorage, make_key, sha256_of
from app.repositories.document_repository import DocumentRepository
from app.repositories.models import Document, Paper
from app.repositories.paper_repository import PaperRepository
from app.services.fulltext_fetcher import FullTextFetcher

pytestmark = pytest.mark.integration

PDF = b"%PDF-1.7 triceps full text"
XML = b"<?xml version='1.0'?><article>triceps</article>"


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


class FakeFinder:
    def __init__(self, links):
        self.links, self.calls = links, 0

    async def find_links(self, paper):
        self.calls += 1
        return self.links


class FakeDownloader:
    """url -> bytes, or a DownloadError reason string."""
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    async def download(self, url, expect):
        self.calls.append(url)
        answer = self.answers[url]
        if isinstance(answer, str):
            raise DownloadError(url, answer)
        return Downloaded(data=answer, kind=expect, content_type=f"application/{expect}", final_url=url)


def make_fetcher(session, tmp_path, finder, downloader) -> FullTextFetcher:
    @asynccontextmanager
    async def unit_of_work():
        yield session
        await session.flush()
    return FullTextFetcher(finder, downloader, LocalFileStorage(tmp_path), unit_of_work)


LINKS = [FullTextLink(kind="xml", url="https://epmc/x.xml", via="europepmc"),
         FullTextLink(kind="pdf", url="https://pub/landing", via="metadata"),
         FullTextLink(kind="pdf", url="https://repo/a.pdf", via="unpaywall")]


async def test_first_working_link_wins_and_is_stored(session, tmp_path):
    paper = await PaperRepository(session).add(Paper(title="Triceps"))
    downloader = FakeDownloader({"https://epmc/x.xml": "HTTP 503",
                                 "https://pub/landing": "expected pdf, got an HTML page",
                                 "https://repo/a.pdf": PDF})
    result = await make_fetcher(session, tmp_path, FakeFinder(LINKS), downloader).fetch(paper.id, PaperRecord(source="x", title="T"))

    assert result.via == "unpaywall" and result.kind == "pdf" and result.size_bytes == len(PDF)
    assert result.failures == ["europepmc: HTTP 503", "metadata: expected pdf, got an HTML page"]
    document = await DocumentRepository(session).get(result.document_id)
    assert document.paper_id == paper.id and document.origin == "open_access" and document.status == "stored"
    assert (tmp_path / document.storage_key).read_bytes() == PDF         # the original file is on disk


async def test_already_stored_paper_is_reused_without_downloading(session, tmp_path):
    paper = await PaperRepository(session).add(Paper(title="Triceps"))
    finder, downloader = FakeFinder(LINKS), FakeDownloader({"https://epmc/x.xml": XML})
    fetcher = make_fetcher(session, tmp_path, finder, downloader)
    first = await fetcher.fetch(paper.id, PaperRecord(source="x", title="T"))
    second = await fetcher.fetch(paper.id, PaperRecord(source="x", title="T"))
    assert first.kind == "xml" and not first.reused
    assert second.reused and second.document_id == first.document_id
    assert finder.calls == 1 and len(downloader.calls) == 1             # nothing looked up or downloaded again


async def test_same_bytes_for_two_papers_are_stored_once(session, tmp_path):
    repo = PaperRepository(session)
    a, b = await repo.add(Paper(title="A")), await repo.add(Paper(title="B"))
    links = [FullTextLink(kind="pdf", url="https://repo/a.pdf", via="unpaywall")]
    fetcher = make_fetcher(session, tmp_path, FakeFinder(links), FakeDownloader({"https://repo/a.pdf": PDF}))
    first = await fetcher.fetch(a.id, PaperRecord(source="x", title="A"))
    second = await fetcher.fetch(b.id, PaperRecord(source="x", title="B"))
    assert first.document_id == second.document_id                        # one row ...
    assert len(list(tmp_path.rglob("*.pdf"))) == 1                       # ... one file


async def test_open_access_download_makes_a_matching_upload_public(session, tmp_path):
    paper = await PaperRepository(session).add(Paper(title="Triceps"))
    sha = sha256_of(PDF)
    upload = await DocumentRepository(session).add(Document(
        sha256=sha, storage_key=make_key(sha, "pdf"), content_type="application/pdf",
        size_bytes=len(PDF), origin="upload"))
    links = [FullTextLink(kind="pdf", url="https://repo/a.pdf", via="unpaywall")]
    fetcher = make_fetcher(session, tmp_path, FakeFinder(links), FakeDownloader({"https://repo/a.pdf": PDF}))
    result = await fetcher.fetch(paper.id, PaperRecord(source="x", title="T"))
    assert result.document_id == upload.id
    assert upload.origin == "open_access" and upload.paper_id == paper.id


async def test_no_links_or_all_failing(session, tmp_path):
    paper = await PaperRepository(session).add(Paper(title="Paywalled"))
    none = await make_fetcher(session, tmp_path, FakeFinder([]), FakeDownloader({})) \
        .fetch(paper.id, PaperRecord(source="x", title="T"))
    assert none.document_id is None and none.failures == ["no free full text found"]

    links = [FullTextLink(kind="pdf", url="https://pub/a.pdf", via="metadata")]
    failed = await make_fetcher(session, tmp_path, FakeFinder(links), FakeDownloader({"https://pub/a.pdf": "HTTP 403"})) \
        .fetch(paper.id, PaperRecord(source="x", title="T"))
    assert failed.document_id is None and failed.failures == ["metadata: HTTP 403"]
    assert not list(tmp_path.rglob("*.*"))                               # nothing stored


# ---------- remembering failures (step 4.7) ----------

class FakeMisses:
    def __init__(self):
        self.data = {}

    async def get(self, paper_id):
        return self.data.get(paper_id)

    async def remember(self, paper_id, reason):
        self.data[paper_id] = reason


def fetcher_with_misses(session, tmp_path, finder, downloader, misses) -> FullTextFetcher:
    fetcher = make_fetcher(session, tmp_path, finder, downloader)
    fetcher.misses = misses
    return fetcher


async def test_permanent_failure_is_remembered_and_skipped_next_time(session, tmp_path):
    paper = await PaperRepository(session).add(Paper(title="Paywalled"))
    links = [FullTextLink(kind="pdf", url="https://pub/a.pdf", via="metadata")]
    finder, misses = FakeFinder(links), FakeMisses()
    fetcher = fetcher_with_misses(session, tmp_path, finder, FakeDownloader({"https://pub/a.pdf": "HTTP 403"}), misses)

    first = await fetcher.fetch(paper.id, PaperRecord(source="x", title="T"))
    second = await fetcher.fetch(paper.id, PaperRecord(source="x", title="T"))
    assert misses.data[paper.id] == "metadata: HTTP 403"
    assert second.failures == ["skipped, failed recently: metadata: HTTP 403"]
    assert finder.calls == 1                                              # no second lookup or download

    retried = await fetcher.fetch(paper.id, PaperRecord(source="x", title="T"), retry_failed=True)
    assert finder.calls == 2 and retried.failures == ["metadata: HTTP 403"]


async def test_temporary_failure_is_not_remembered(session, tmp_path):
    paper = await PaperRepository(session).add(Paper(title="Europe PMC down"))
    links = [FullTextLink(kind="xml", url="https://epmc/x.xml", via="europepmc")]
    misses = FakeMisses()
    fetcher = fetcher_with_misses(session, tmp_path, FakeFinder(links),
                                  FakeDownloader({"https://epmc/x.xml": "timed out after 60s"}), misses)
    await fetcher.fetch(paper.id, PaperRecord(source="x", title="T"))
    assert misses.data == {}                                              # will be tried again next run
