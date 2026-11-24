"""
Extraction service (step 4.4): stored file -> sections in the database.

    read the file from FileStorage -> JATS or PDF extractor (in a worker thread: parsing is CPU work)
    -> document_sections rows -> documents.status = extracted (or failed + reason)
    -> the paper's fulltext_status = "full"

Already extracted? Reused (pass force=True to redo it, e.g. after improving the extractor;
the original file is stored, so no download is needed).
"""

import asyncio

from app.domain.errors import ExtractionError, NotFoundError
from app.domain.schemas import ExtractionResult
from app.integrations.extraction.jats import extract_jats
from app.integrations.extraction.pdf import extract_pdf
from app.integrations.extraction.sections import Extraction
from app.integrations.storage.file_storage import FileStorage
from app.repositories.document_repository import DocumentRepository
from app.repositories.paper_repository import PaperRepository
from app.services.fulltext_fetcher import UnitOfWork


class ExtractionService:
    def __init__(self, storage: FileStorage, unit_of_work: UnitOfWork, *, max_pages: int = 100):
        self.storage = storage
        self.unit_of_work = unit_of_work
        self.max_pages = max_pages

    async def extract(self, document_id: int, *, force: bool = False) -> ExtractionResult:
        async with self.unit_of_work() as session:
            documents = DocumentRepository(session)
            document = await documents.get(document_id)
            if document is None:
                raise NotFoundError(f"document {document_id} does not exist")
            if document.status == "extracted" and not force:
                return _result(document_id, await documents.sections(document_id), reused=True)
            key, content_type = document.storage_key, document.content_type

        data = await self.storage.read(key)
        try:
            extraction = await asyncio.to_thread(self._parse, content_type, data)
        except ExtractionError as e:
            async with self.unit_of_work() as session:
                document = await DocumentRepository(session).get(document_id)
                document.status, document.error = "failed", str(e)
            return ExtractionResult(document_id=document_id, status="failed", error=str(e))

        async with self.unit_of_work() as session:
            documents = DocumentRepository(session)
            await documents.replace_sections(document_id, [(s.kind, s.title, s.text) for s in extraction.sections])
            document = await documents.get(document_id)
            document.status, document.error, document.page_count = "extracted", None, extraction.page_count
            if document.paper_id is not None:
                paper = await PaperRepository(session).get(document.paper_id)
                paper.fulltext_status = "full"
            return _result(document_id, extraction.sections)

    def _parse(self, content_type: str, data: bytes) -> Extraction:
        if content_type == "application/xml":
            return extract_jats(data)
        if content_type == "application/pdf":
            return extract_pdf(data, max_pages=self.max_pages)
        raise ExtractionError(f"unsupported file type {content_type}")


def _result(document_id: int, sections, reused: bool = False) -> ExtractionResult:
    return ExtractionResult(document_id=document_id, status="extracted", reused=reused, sections=len(sections),
                            characters=sum(len(s.text) for s in sections),
                            kinds=sorted({s.kind for s in sections}))
