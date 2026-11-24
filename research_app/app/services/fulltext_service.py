"""
Full text for a set of papers (step 4.5): fetch (4.3) -> extract (4.4) -> PaperText (context + source).
Papers without a free full text are simply missing from the result: they are summarised from the abstract.
"""

from app.common.logging import get_logger
from app.domain.errors import AppError
from app.domain.schemas import PaperRecord
from app.repositories.document_repository import DocumentRepository
from app.services.extraction_service import ExtractionService
from app.services.fulltext_context import PaperText, paper_text
from app.services.fulltext_fetcher import FullTextFetcher, UnitOfWork

log = get_logger(__name__)


class FullTextService:
    def __init__(self, fetcher: FullTextFetcher, extractor: ExtractionService, unit_of_work: UnitOfWork):
        self.fetcher = fetcher
        self.extractor = extractor
        self.unit_of_work = unit_of_work

    async def texts_for(self, papers: dict[int, PaperRecord]) -> dict[int, PaperText]:
        """{paper_id: paper} -> {paper_id: PaperText} for the papers whose full text we could get."""
        fetched = await self.fetcher.fetch_many(papers)
        texts: dict[int, PaperText] = {}
        for paper_id, fetch in fetched.items():
            if fetch.document_id is None:
                continue                                         # paywalled / not found: abstract only
            try:
                extraction = await self.extractor.extract(fetch.document_id)   # one at a time: CPU work
                if extraction.status != "extracted":
                    continue
                async with self.unit_of_work() as session:
                    sections = await DocumentRepository(session).sections(fetch.document_id)
            except (AppError, OSError) as e:                     # e.g. the stored file went missing
                log.warning("full text unusable for paper %d: %s", paper_id, e)
                continue
            texts[paper_id] = paper_text([(s.kind, s.title, s.text) for s in sections])
        return texts
