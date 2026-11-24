"""
Europe PMC: medicine, biology, health, sports science. Includes all of PubMed,
plus abstracts, citation counts and links to free full texts (PubMed Central).
Docs: https://europepmc.org/RestfulWebService
"""

import re

import httpx

from app.domain.schemas import PaperRecord
from app.integrations.sources.base import http_get, normalize_doi, parse_date, strip_html

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
REST_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"


class EuropePmcClient:
    name = "europepmc"

    def __init__(self, client: httpx.AsyncClient, timeout: float = 30):
        self.client = client          # the SHARED async HTTP client
        self.timeout = timeout

    async def search(self, query: str, limit: int = 15) -> list[PaperRecord]:
        # TITLE_ABS: match words in the title/abstract only. Without it Europe PMC also searches
        # full texts, so a paper mentioning "triceps" once in its methods would match (tested: 1262 vs 44 hits).
        params = {
            "query": f"TITLE_ABS:({query})",
            "format": "json",
            "resultType": "core",        # "core" includes abstracts, authors, links (not just ids)
            "pageSize": min(limit, 100),
        }
        response = await http_get(self.client, self.name, BASE_URL, params=params, timeout=self.timeout)
        results = (response.json().get("resultList") or {}).get("result", [])
        records = [to_record(item, query) for item in results]
        return [r for r in records if r is not None]

    async def find_pmcid(self, *, pmid: str | None = None, doi: str | None = None) -> str | None:
        """
        The PubMed Central id (e.g. 'PMC8174684') if Europe PMC has this paper's open-access full text.
        We store PMIDs but not PMC ids, so look the paper up (one small 'lite' search, no retries).
        """
        if pmid and pmid.isdigit():
            query = f"EXT_ID:{pmid} AND SRC:MED"
        elif doi:
            query = f'DOI:"{doi.replace(chr(34), "")}"'          # strip quotes: they'd break the query
        else:
            return None
        response = await http_get(self.client, self.name, f"{REST_URL}/search", timeout=self.timeout, retries=0,
                                  params={"query": query, "format": "json", "resultType": "lite", "pageSize": 1})
        results = (response.json().get("resultList") or {}).get("result", [])
        if not results:
            return None
        item = results[0]
        pmcid = item.get("pmcid") or ""
        # isOpenAccess: the licence allows reuse; inEPMC: Europe PMC really has the full text
        if item.get("isOpenAccess") == "Y" and item.get("inEPMC") == "Y" and re.fullmatch(r"PMC\d+", pmcid):
            return pmcid
        return None


def to_record(item: dict, query: str) -> PaperRecord | None:
    """Convert ONE Europe PMC result into our common PaperRecord shape."""
    title = (item.get("title") or "").strip().rstrip(".")      # Europe PMC titles end with "."
    if not title:
        return None

    journal = ((item.get("journalInfo") or {}).get("journal") or {}).get("title")
    pub_types = [t.lower() for t in (item.get("pubTypeList") or {}).get("pubType", [])]
    published = parse_date(item.get("firstPublicationDate"))

    # pubYear is a string like "2024" (sometimes missing) → fall back to the publication date
    year = int(item["pubYear"]) if str(item.get("pubYear") or "").isdigit() else (
        published.year if published else None)

    # full author names if available, otherwise split the short "Lee A, Kim B." string
    authors = [a.get("fullName") for a in (item.get("authorList") or {}).get("author", [])
               if a.get("fullName")]
    if not authors and item.get("authorString"):
        authors = [a.strip() for a in item["authorString"].rstrip(".").split(",") if a.strip()]

    return PaperRecord(
        source="europepmc",
        title=title,
        doi=normalize_doi(item.get("doi")),
        pmid=item.get("pmid"),
        authors=authors,
        venue=journal,
        venue_type="preprint" if item.get("source") == "PPR" else ("journal" if journal else None),
        publication_type=_publication_type(pub_types, item.get("source")),
        pub_types=pub_types,                      # kept whole: "randomized controlled trial" checks the LLM (6.7)
        year=year,
        published_date=published,
        abstract=strip_html(item.get("abstractText")),
        citation_count=item.get("citedByCount"),
        is_open_access=item.get("isOpenAccess") == "Y",
        pdf_url=_pdf_url(item),
        landing_url=f"https://europepmc.org/article/{item.get('source')}/{item.get('id')}",
        found_by_query=query,
    )


def _publication_type(pub_types: list[str], source: str | None) -> str:
    """Europe PMC gives a list like ['Review', 'Journal Article'] → one simple type."""
    if source == "PPR" or "preprint" in pub_types:       # PPR = Europe PMC's code for preprints
        return "preprint"
    if any("review" in t for t in pub_types):
        return "review"
    return "article"


def _pdf_url(item: dict) -> str | None:
    """First free/open-access PDF link, if Europe PMC lists one."""
    for link in (item.get("fullTextUrlList") or {}).get("fullTextUrl", []):
        if link.get("documentStyle") == "pdf" and link.get("availability") in ("Open access", "Free"):
            return link.get("url")
    return None
