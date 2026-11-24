"""
Full-text fetcher (step 4.3): get ONE stored copy of a paper's full text.

    already stored for this paper?  -> reuse it (no download at all)
    failed permanently recently?    -> skip it (remembered in Redis for 24 h, step 4.7)
    else: links = finder.find_links(paper)            (step 4.2, best first)
          try each link with the SafeDownloader       (first one that works wins)
          fingerprint (SHA-256) -> save to FileStorage -> row in `documents`

Same bytes = same fingerprint = stored once, even if two papers or two users lead to the same file.
The file is saved BEFORE the database row: if the row fails, the file is just an unused copy that
the next attempt reuses (same name). The other order could leave a row pointing at no file.
"""

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from app.common.logging import get_logger
from app.domain.errors import DownloadError
from app.domain.schemas import FetchResult, FullTextLink, PaperRecord
from app.integrations.downloader import Downloaded, Kind
from app.integrations.miss_cache import is_permanent
from app.integrations.storage.file_storage import FileStorage, make_key, sha256_of
from app.repositories.document_repository import DocumentRepository
from app.repositories.models import Document

log = get_logger(__name__)
UnitOfWork = Callable[[], AbstractAsyncContextManager[AsyncSession]]


class LinkFinder(Protocol):
    async def find_links(self, paper: PaperRecord) -> list[FullTextLink]: ...


class Downloader(Protocol):
    async def download(self, url: str, expect: Kind) -> Downloaded: ...


class Misses(Protocol):                        # MissCache (Redis) or a fake in tests
    async def get(self, paper_id: int) -> str | None: ...
    async def remember(self, paper_id: int, reason: str) -> None: ...


class FullTextFetcher:
    def __init__(self, finder: LinkFinder, downloader: Downloader, storage: FileStorage,
                 unit_of_work: UnitOfWork, *, max_parallel: int = 3, misses: Misses | None = None):
        self.finder = finder
        self.downloader = downloader
        self.storage = storage
        self.unit_of_work = unit_of_work
        self.max_parallel = max_parallel          # downloads at the same time (be polite to publishers)
        self.misses = misses                      # remembers papers with no full text (step 4.7)

    async def fetch(self, paper_id: int, paper: PaperRecord, *, retry_failed: bool = False) -> FetchResult:
        existing = await self._existing(paper_id)
        if existing is not None:
            return existing
        if self.misses is not None and not retry_failed:
            reason = await self.misses.get(paper_id)
            if reason is not None:                # failed permanently recently: don't try again yet
                return FetchResult(paper_id=paper_id, failures=[f"skipped, failed recently: {reason}"])

        failures: list[str] = []
        for link in await self.finder.find_links(paper):
            try:
                downloaded = await self.downloader.download(link.url, link.kind)
            except DownloadError as e:
                failures.append(f"{link.via}: {e.reason}")
                continue
            document_id = await self._store(paper_id, downloaded)
            return FetchResult(paper_id=paper_id, document_id=document_id, kind=downloaded.kind, via=link.via,
                               url=downloaded.final_url, size_bytes=len(downloaded.data), failures=failures)
        if not failures:
            failures.append("no free full text found")
        if self.misses is not None and is_permanent(failures):
            await self.misses.remember(paper_id, "; ".join(failures))
        return FetchResult(paper_id=paper_id, failures=failures)

    async def fetch_many(self, papers: dict[int, PaperRecord], *, retry_failed: bool = False
                         ) -> dict[int, FetchResult]:
        """fetch() for several papers ({paper_id: paper}), a few downloads at a time."""
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def one(paper_id: int, paper: PaperRecord) -> tuple[int, FetchResult]:
            async with semaphore:
                return paper_id, await self.fetch(paper_id, paper, retry_failed=retry_failed)

        return dict(await asyncio.gather(*(one(pid, p) for pid, p in papers.items())))

    async def _existing(self, paper_id: int) -> FetchResult | None:
        """A stored, usable document for this paper (XML preferred over PDF), or None."""
        async with self.unit_of_work() as session:
            documents = [d for d in await DocumentRepository(session).for_paper(paper_id) if d.status != "failed"]
        if not documents:
            return None
        best = min(documents, key=lambda d: 0 if d.content_type == "application/xml" else 1)
        return FetchResult(paper_id=paper_id, document_id=best.id,
                           kind="xml" if best.content_type == "application/xml" else "pdf",
                           url=best.source_url, size_bytes=best.size_bytes, reused=True)

    async def _store(self, paper_id: int, downloaded: Downloaded) -> int:
        sha = sha256_of(downloaded.data)
        key = make_key(sha, downloaded.kind)
        await self.storage.save(key, downloaded.data)               # file first (see module docstring)
        async with self.unit_of_work() as session:
            documents = DocumentRepository(session)
            document = await documents.get_by_sha256(sha)
            if document is None:
                document = await documents.add(Document(
                    sha256=sha, storage_key=key, content_type=downloaded.content_type,
                    size_bytes=len(downloaded.data), origin="open_access", source_url=downloaded.final_url,
                    paper_id=paper_id, status="stored"))
            else:
                if document.paper_id is None:                        # e.g. an upload we can now attach
                    document.paper_id = paper_id
                if document.origin == "upload":                      # the same bytes are openly available,
                    document.origin = "open_access"                  # so the file is no longer private
            log.info("stored full text for paper %d: %s, %d KB", paper_id, downloaded.kind,
                     len(downloaded.data) // 1024)
            return document.id
