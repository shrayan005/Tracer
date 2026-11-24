"""
Unpaywall: finds LEGAL free copies of a paper by its DOI (publisher sites, university repositories,
PubMed Central...). Free API; every request must include a contact email (their rule).
Docs: https://unpaywall.org/products/api
"""

import re
from dataclasses import dataclass, field
from urllib.parse import quote

import httpx

from app.domain.errors import SourceError
from app.integrations.sources.base import http_get

BASE_URL = "https://api.unpaywall.org/v2"
PMCID = re.compile(r"PMC\d{4,9}")


@dataclass
class UnpaywallResult:
    pdfs: list[tuple[str, str | None]] = field(default_factory=list)   # (pdf url, license), best first
    pmcid: str | None = None          # PubMed Central id, if any free copy lives there


class UnpaywallClient:
    name = "unpaywall"

    def __init__(self, client: httpx.AsyncClient, email: str, timeout: float = 30):
        self.client = client          # the SHARED async HTTP client
        self.email = email
        self.timeout = timeout

    async def lookup(self, doi: str) -> UnpaywallResult:
        """Free copies Unpaywall knows for a DOI. Empty result if none (or the DOI is unknown)."""
        url = f"{BASE_URL}/{quote(doi, safe='/')}"
        try:
            response = await http_get(self.client, self.name, url, params={"email": self.email},
                                      timeout=self.timeout, retries=0)
        except SourceError as e:
            if e.status == 404:                      # DOI unknown to Unpaywall: not an error for us
                return UnpaywallResult()
            raise
        return parse(response.json())


def parse(data: dict) -> UnpaywallResult:
    """Turn Unpaywall's JSON into PDF links + a PMC id (pure function: easy to test)."""
    result = UnpaywallResult()
    if not data.get("is_oa"):
        return result
    locations = [data.get("best_oa_location")] + list(data.get("oa_locations") or [])
    for location in filter(None, locations):
        pdf = location.get("url_for_pdf")
        if pdf and pdf not in [url for url, _ in result.pdfs]:
            result.pdfs.append((pdf, location.get("license")))
        for url in (pdf, location.get("url"), location.get("url_for_landing_page")):
            if result.pmcid is None and url and (match := PMCID.search(url)):
                result.pmcid = match.group(0)        # e.g. .../articles/PMC8174684/
    return result
