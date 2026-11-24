"""
OpenAlex: free catalogue of ~all scholarly works, every field. Our universal backbone.
Docs: https://docs.openalex.org   Free API key (more daily budget): https://openalex.org/settings/api
"""

import httpx

from app.domain.schemas import PaperRecord
from app.integrations.sources.base import http_get, last_path_part, normalize_doi, parse_date

BASE_URL = "https://api.openalex.org/works"


class OpenAlexClient:
    name = "openalex"

    def __init__(self, client: httpx.AsyncClient, api_key: str | None = None, timeout: float = 30):
        self.client = client          # the SHARED async HTTP client (connection pool)
        self.api_key = api_key
        self.timeout = timeout

    async def search(self, query: str, limit: int = 15) -> list[PaperRecord]:
        params = {
            "search": query,
            "per-page": min(limit, 50),
            "filter": "type:article|review|preprint",   # skip book chapters, datasets, etc.
        }
        if self.api_key:
            params["api_key"] = self.api_key
        response = await http_get(self.client, self.name, BASE_URL, params=params, timeout=self.timeout)
        data = response.json()
        records = [to_record(work, query) for work in data.get("results", [])]
        return [r for r in records if r is not None]     # drop results without a title

    async def references(self, openalex_id: str) -> list[PaperRecord]:
        """The works a paper cites (its reference list), as PaperRecords. Used by the search-recall eval:
        a systematic review's references are a free, expert-made list of the studies on its topic."""
        response = await http_get(self.client, self.name, f"{BASE_URL}/{openalex_id}", timeout=self.timeout,
                                  params={"select": "id,referenced_works", **self._key()})
        ids = [last_path_part(w) for w in response.json().get("referenced_works") or []]
        return await self.works([i for i in ids if i])

    async def works(self, openalex_ids: list[str]) -> list[PaperRecord]:
        """Look up many works by OpenAlex id (50 per request, the API's limit for an OR filter)."""
        records = []
        for start in range(0, len(openalex_ids), 50):
            batch = openalex_ids[start:start + 50]
            response = await http_get(self.client, self.name, BASE_URL, timeout=self.timeout, params={
                "filter": "openalex_id:" + "|".join(batch), "per-page": 50, **self._key()})
            records += [r for r in (to_record(w, "") for w in response.json().get("results", [])) if r]
        return records

    def _key(self) -> dict:
        return {"api_key": self.api_key} if self.api_key else {}


def to_record(work: dict, query: str) -> PaperRecord | None:
    """Convert ONE OpenAlex result into our common PaperRecord shape."""
    title = work.get("display_name") or work.get("title")
    if not title:
        return None
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    open_access = work.get("open_access") or {}
    ids = work.get("ids") or {}
    return PaperRecord(
        source="openalex",
        title=title.strip(),
        doi=normalize_doi(work.get("doi")),
        openalex_id=last_path_part(work.get("id")),
        pmid=last_path_part(ids.get("pmid")),
        authors=[a["author"]["display_name"] for a in work.get("authorships", [])
                 if a.get("author", {}).get("display_name")],
        venue=source.get("display_name"),
        venue_type=source.get("type"),               # journal / repository / conference ...
        publication_type=work.get("type"),           # article / review / preprint
        year=work.get("publication_year"),
        published_date=parse_date(work.get("publication_date")),
        abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
        citation_count=work.get("cited_by_count"),
        is_open_access=open_access.get("is_oa"),
        is_retracted=bool(work.get("is_retracted")),
        pdf_url=location.get("pdf_url"),
        landing_url=location.get("landing_page_url") or open_access.get("oa_url"),
        found_by_query=query,
    )


def reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """
    OpenAlex stores abstracts as {word: [positions]} for legal reasons, e.g.
        {"Muscle": [0], "activation": [1, 3], "increased": [2]}
    Put every word back at its position to rebuild the text:
        "Muscle activation increased activation"
    """
    if not inverted_index:
        return None
    positions = [(pos, word) for word, places in inverted_index.items() for pos in places]
    return " ".join(word for _, word in sorted(positions)) or None

