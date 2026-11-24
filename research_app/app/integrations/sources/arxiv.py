"""
arXiv: free preprints in computer science, physics, maths, statistics.
Returns XML (Atom), not JSON. arXiv asks clients to wait ~3 seconds between requests,
so all calls share one asyncio.Lock and a spacing timer, even when searches run in parallel.
Docs: https://info.arxiv.org/help/api/user-manual.html
"""

import asyncio
import re
import time
import xml.etree.ElementTree as ET

import httpx

from app.domain.schemas import PaperRecord
from app.integrations.sources.base import http_get, normalize_doi, parse_date

BASE_URL = "https://export.arxiv.org/api/query"
# XML "namespaces": Atom tags and arXiv's own extra tags have different prefixes
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
MIN_SECONDS_BETWEEN_CALLS = 3.0


class ArxivClient:
    name = "arxiv"

    def __init__(self, client: httpx.AsyncClient, timeout: float = 30):
        self.client = client
        self.timeout = timeout
        self._lock = asyncio.Lock()     # only ONE arXiv request at a time
        self._last_call = 0.0           # when the previous request finished

    async def search(self, query: str, limit: int = 15) -> list[PaperRecord]:
        words = re.findall(r"[A-Za-z0-9\-]+", query)     # keep only safe words
        if not words:
            return []
        search_query = " AND ".join(f"all:{w}" for w in words)   # every word must appear

        async with self._lock:          # other searches wait here until it's their turn
            wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)                # polite spacing, without blocking others
            response = await http_get(self.client, self.name, BASE_URL, timeout=self.timeout, params={
                "search_query": search_query, "max_results": min(limit, 50), "sortBy": "relevance"})
            self._last_call = time.monotonic()
        return parse_feed(response.text, query)


def parse_feed(xml_text: str, query: str) -> list[PaperRecord]:
    """Convert arXiv's Atom XML into PaperRecords (plain function: easy to test)."""
    root = ET.fromstring(xml_text)
    records = []
    for entry in root.findall("a:entry", NS):                  # one <entry> per paper
        title = " ".join(entry.findtext("a:title", default="", namespaces=NS).split())
        if not title:
            continue
        abs_url = entry.findtext("a:id", default="", namespaces=NS)        # http://arxiv.org/abs/2209.15001v3
        arxiv_id = re.sub(r"v\d+$", "", abs_url.rsplit("/abs/", 1)[-1]) or None   # → 2209.15001
        published = parse_date(entry.findtext("a:published", namespaces=NS))
        pdf = next((link.get("href") for link in entry.findall("a:link", NS)
                    if link.get("title") == "pdf"), None)
        records.append(PaperRecord(
            source="arxiv",
            title=title,
            doi=normalize_doi(entry.findtext("arxiv:doi", namespaces=NS)),
            arxiv_id=arxiv_id,
            authors=[a.findtext("a:name", default="", namespaces=NS) for a in entry.findall("a:author", NS)],
            venue=entry.findtext("arxiv:journal_ref", namespaces=NS) or "arXiv",
            venue_type="preprint",
            publication_type="preprint",
            year=published.year if published else None,
            published_date=published,
            abstract=" ".join(entry.findtext("a:summary", default="", namespaces=NS).split()) or None,
            is_open_access=True,           # every arXiv paper is free to read
            pdf_url=pdf,
            landing_url=abs_url or None,
            found_by_query=query,
        ))
    return records
