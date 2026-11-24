"""
Semantic Scholar: all fields (strongest in CS and biomedicine). Extras: influential
citation counts and short AI summaries (TLDR).
Without an API key the shared pool is often overloaded (429), so this source is OPTIONAL:
it is only used when SEMANTIC_SCHOLAR_API_KEY is set (free: semanticscholar.org/product/api).

Rate limit: a key allows about 1 request per second. The discovery service runs up to 6
searches at once, so this client spaces its own calls with a lock (same idea as arXiv).
Tested: without the spacing, parallel queries got HTTP 429. Even WITH spacing, the search
endpoint sometimes returns 429 for a few seconds (throttling on their side), so a 429 gets
one retry. It stays a best-effort source: if it still fails, discovery just skips it.
"""

import asyncio
import time

import httpx

from app.domain.schemas import PaperRecord
from app.integrations.sources.base import http_get, normalize_doi, parse_date

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
# Semantic Scholar only returns the fields you ask for
FIELDS = ("title,abstract,year,venue,externalIds,citationCount,influentialCitationCount,"
          "isOpenAccess,openAccessPdf,publicationDate,authors,publicationTypes,journal,tldr,url")
MIN_SECONDS_BETWEEN_CALLS = 1.1   # keyed limit is ~1 request/second; a little margin
RETRIES = 1                       # one retry after 2 s (measured: more retries mostly failed anyway
                                  # and made each query take ~13 s)


class SemanticScholarClient:
    name = "semantic_scholar"

    def __init__(self, client: httpx.AsyncClient, api_key: str | None = None, timeout: float = 30):
        self.client = client
        self.api_key = api_key
        self.timeout = timeout
        self._lock = asyncio.Lock()     # only ONE Semantic Scholar request at a time
        self._last_call = 0.0           # when the previous request finished

    async def search(self, query: str, limit: int = 15) -> list[PaperRecord]:
        headers = {"x-api-key": self.api_key} if self.api_key else None   # key goes in a HEADER here
        async with self._lock:          # other searches wait here (without blocking the event loop)
            wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            response = await http_get(self.client, self.name, BASE_URL, headers=headers,
                                      timeout=self.timeout, retries=RETRIES,
                                      params={"query": query, "limit": min(limit, 100), "fields": FIELDS})
            self._last_call = time.monotonic()
        records = [to_record(p, query) for p in response.json().get("data", [])]
        return [r for r in records if r is not None]


def to_record(paper: dict, query: str) -> PaperRecord | None:
    """Convert ONE Semantic Scholar result into our common PaperRecord shape."""
    if not paper.get("title"):
        return None
    ids = paper.get("externalIds") or {}                 # {"DOI": ..., "PubMed": ..., "ArXiv": ...}
    types = [t.lower() for t in (paper.get("publicationTypes") or [])]
    pdf = (paper.get("openAccessPdf") or {}).get("url")
    return PaperRecord(
        source="semantic_scholar",
        title=paper["title"].strip(),
        doi=normalize_doi(ids.get("DOI")),
        s2_id=paper.get("paperId"),
        pmid=str(ids["PubMed"]) if ids.get("PubMed") else None,     # sometimes a number → text
        arxiv_id=ids.get("ArXiv"),
        authors=[a["name"] for a in paper.get("authors") or [] if a.get("name")],
        venue=paper.get("venue") or ((paper.get("journal") or {}).get("name")),
        venue_type="journal" if "journalarticle" in types else (
            "conference" if "conference" in types else None),
        publication_type="review" if "review" in types else "article",
        pub_types=types,                          # e.g. "metaanalysis", "clinicaltrial", "review"
        year=paper.get("year"),
        published_date=parse_date(paper.get("publicationDate")),
        abstract=paper.get("abstract"),
        tldr=(paper.get("tldr") or {}).get("text"),
        citation_count=paper.get("citationCount"),
        influential_citation_count=paper.get("influentialCitationCount"),
        is_open_access=paper.get("isOpenAccess"),
        pdf_url=pdf or None,                                # "" (empty) → None
        landing_url=paper.get("url"),
        found_by_query=query,
    )
