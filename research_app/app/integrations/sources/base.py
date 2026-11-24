"""
Shared pieces for all academic source clients.

Every source (OpenAlex, Europe PMC, arXiv, Semantic Scholar) returns papers in its
own format. Each client converts them into ONE common shape: PaperRecord.

Async: all clients share ONE httpx.AsyncClient (a connection pool), created in
dependencies.py and closed in close_resources().
"""

import asyncio
import re
from datetime import date
from typing import Protocol

import httpx

from app.domain.errors import SourceError
from app.domain.schemas import PaperRecord

USER_AGENT = "research-app-learning/0.1"


class SourceClient(Protocol):
    """Anything with a name and an async search() method is a source (real or fake)."""
    name: str

    async def search(self, query: str, limit: int) -> list[PaperRecord]: ...


async def http_get(client: httpx.AsyncClient, source: str, url: str, *, params: dict | None = None,
                   headers: dict | None = None, timeout: float = 30, retries: int = 1) -> httpx.Response:
    """GET with a timeout and one polite retry on 429/5xx. Raises SourceError on failure."""
    all_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    for attempt in range(retries + 1):
        try:
            response = await client.get(url, params=params, headers=all_headers, timeout=timeout)
        except httpx.HTTPError as e:                       # no internet, DNS failure, timeout...
            if attempt < retries:
                await asyncio.sleep(2)
                continue
            raise SourceError(source, f"network error: {e}", retryable=True) from e

        if response.status_code == 429 or response.status_code >= 500:   # busy / server problem
            if attempt < retries:
                await asyncio.sleep(2 * (attempt + 1))
                continue
            raise SourceError(source, f"HTTP {response.status_code}",
                              status=response.status_code, retryable=True)
        if response.status_code >= 400:                                   # our request was wrong
            raise SourceError(source, f"HTTP {response.status_code}: {response.text[:200]}",
                              status=response.status_code)
        return response
    raise SourceError(source, "unreachable")   # the loop always returns or raises before this


# ---------- small clean-up helpers (no waiting → normal functions) ----------

def normalize_doi(doi: str | None) -> str | None:
    """'https://doi.org/10.1000/ABC' -> '10.1000/abc' (DOIs are case-insensitive)."""
    if not doi:
        return None
    doi = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi or None


def last_path_part(url_or_id: str | None) -> str | None:
    """'https://openalex.org/W123' -> 'W123', 'https://pubmed.ncbi.nlm.nih.gov/2129879' -> '2129879'."""
    if not url_or_id:
        return None
    return url_or_id.rstrip("/").rsplit("/", 1)[-1] or None


def parse_date(value: str | None) -> date | None:
    """'2023-03-15' or '2023-03-15T10:00:00Z' -> date(2023, 3, 15); anything else -> None."""
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def strip_html(text: str | None) -> str | None:
    """Abstracts sometimes contain <i>, <sup> tags; remove them."""
    if not text:
        return None
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip() or None
