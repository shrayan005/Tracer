"""Data access for the papers table."""

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.schemas import PaperRecord
from app.repositories.models import Paper

ID_FIELDS = ("doi", "openalex_id", "s2_id", "pmid", "arxiv_id")   # in priority order: DOI first
# Fields copied from a PaperRecord when the stored paper doesn't have them yet
FILLABLE_FIELDS = ("title", "venue", "venue_type", "publication_type", "year", "published_date",
                   "abstract", "tldr", "is_open_access")


class PaperRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, paper_id: int) -> Paper | None:
        return await self.session.get(Paper, paper_id)

    async def get_by_doi(self, doi: str) -> Paper | None:
        return await self.session.scalar(select(Paper).where(Paper.doi == doi.lower()))

    async def find_by_any_id(self, **ids: str | None) -> Paper | None:
        """Find a paper matching ANY of the given identifiers (doi, openalex_id, ...)."""
        conditions = [getattr(Paper, name) == value for name, value in ids.items()
                      if name in ID_FIELDS and value]
        if not conditions:
            return None
        return await self.session.scalar(select(Paper).where(or_(*conditions)).limit(1))

    async def add(self, paper: Paper) -> Paper:
        if paper.doi:
            paper.doi = paper.doi.lower()   # DOIs are case-insensitive; store one form
        self.session.add(paper)
        await self.session.flush()
        return paper

    async def upsert_from_record(self, record: PaperRecord) -> Paper:
        """
        Insert the paper, or update the stored one if we already know it by any id.
        Existing values are kept; only missing values are filled in. Counts take the maximum.
        """
        paper = await self._find_existing(record)

        if paper is None:                                              # NEW paper → insert
            paper = Paper(title=record.title, authors=record.authors, citation_count=record.citation_count,
                          influential_citation_count=record.influential_citation_count,
                          is_retracted=record.is_retracted, source_urls={}, pub_types=list(record.pub_types))
            for name in ID_FIELDS + FILLABLE_FIELDS:
                setattr(paper, name, getattr(record, name))
            self._set_urls(paper, record)
            return await self.add(paper)

        # KNOWN paper → fill in what's missing
        for name in ID_FIELDS:
            value = getattr(record, name)
            # fill a missing id, unless ANOTHER stored paper already owns it (unique constraint)
            if getattr(paper, name) is None and value:
                owner = await self.find_by_any_id(**{name: value})
                if owner is None or owner.id == paper.id:
                    setattr(paper, name, value)
        for name in FILLABLE_FIELDS:
            if getattr(paper, name) in (None, "") and getattr(record, name) not in (None, ""):
                setattr(paper, name, getattr(record, name))
        if not paper.authors and record.authors:
            paper.authors = record.authors
        paper.citation_count = _max(paper.citation_count, record.citation_count)
        paper.influential_citation_count = _max(paper.influential_citation_count,
                                                record.influential_citation_count)
        paper.is_retracted = paper.is_retracted or record.is_retracted
        new_tags = [t for t in record.pub_types if t not in (paper.pub_types or [])]
        if new_tags:
            paper.pub_types = list(paper.pub_types or []) + new_tags      # JSONB: assign a new list
        self._set_urls(paper, record)
        await self.session.flush()
        return paper

    async def _find_existing(self, record: PaperRecord) -> Paper | None:
        """
        Look ids up ONE AT A TIME in priority order (DOI is the most reliable).
        A record could match two different stored papers (one by DOI, another by PubMed id);
        checking in order makes the choice predictable instead of random.
        """
        for name in ID_FIELDS:
            value = getattr(record, name)
            if value:
                paper = await self.find_by_any_id(**{name: value})
                if paper is not None:
                    return paper
        return None

    @staticmethod
    def _set_urls(paper: Paper, record: PaperRecord) -> None:
        urls = dict(paper.source_urls or {})          # copy: JSONB changes must be reassigned
        if record.pdf_url and "pdf" not in urls:
            urls["pdf"] = record.pdf_url
        if record.landing_url:
            urls.setdefault(record.source, record.landing_url)
        paper.source_urls = urls


def to_record(paper: Paper) -> PaperRecord:
    """A stored paper back in the common PaperRecord shape (so services and citations can use it)."""
    urls = paper.source_urls or {}
    landing = next((url for key, url in urls.items() if key != "pdf"), None)
    return PaperRecord(
        source="db", title=paper.title, doi=paper.doi, openalex_id=paper.openalex_id, s2_id=paper.s2_id,
        pmid=paper.pmid, arxiv_id=paper.arxiv_id, authors=paper.authors or [], venue=paper.venue,
        venue_type=paper.venue_type, publication_type=paper.publication_type, pub_types=paper.pub_types or [],
        year=paper.year,
        published_date=paper.published_date, abstract=paper.abstract, tldr=paper.tldr,
        citation_count=paper.citation_count, influential_citation_count=paper.influential_citation_count,
        is_open_access=paper.is_open_access, is_retracted=paper.is_retracted,
        pdf_url=urls.get("pdf"), landing_url=landing,
    )


def _max(a: int | None, b: int | None) -> int | None:
    values = [v for v in (a, b) if v is not None]
    return max(values) if values else None
