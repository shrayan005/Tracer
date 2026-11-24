"""
Full-text finder (step 4.2): WHERE might a legal free full text of this paper be?
It only finds links; downloading is step 4.3. Returns links best first:

  1. Europe PMC XML   structured (JATS): sections and tables already marked up -> best quality
  2. arXiv PDF        if the paper has an arXiv id (no request needed)
  3. metadata PDF     the pdf_url a search source already gave us (no request needed)
  4. Unpaywall PDFs   legal copies on publisher sites and university repositories

The Europe PMC XML URL needs a PMC id. Two ways to get it, run at the same time:
  - ask Europe PMC (by PMID/DOI)            - often slow or down (measured: 5-33 s, timeouts)
  - spot it in an Unpaywall/metadata link   - free: ".../articles/PMC8174684/" names it
Every lookup has a time limit, and a failing source is skipped: fewer links, never a crash.
"""

import asyncio
import re
from typing import Protocol

from app.common.logging import get_logger
from app.domain.errors import SourceError
from app.domain.schemas import FullTextLink, PaperRecord
from app.integrations.sources.unpaywall import UnpaywallResult

log = get_logger(__name__)

EPMC_REST = "https://www.ebi.ac.uk/europepmc/webservices/rest"
PMCID = re.compile(r"PMC\d{4,9}")


class PmcLookup(Protocol):                     # EuropePmcClient (or a fake in tests)
    async def find_pmcid(self, *, pmid: str | None = None, doi: str | None = None) -> str | None: ...


class OpenAccessLookup(Protocol):              # UnpaywallClient (or a fake in tests)
    async def lookup(self, doi: str) -> UnpaywallResult: ...


class FullTextFinder:
    def __init__(self, europepmc: PmcLookup | None, unpaywall: OpenAccessLookup | None, *,
                 max_parallel: int = 10, lookup_timeout_s: float = 10.0, pmc_xml: bool = True):
        self.europepmc = europepmc
        self.pmc_xml = pmc_xml                      # False: never download XML from Europe PMC (PDFs only)
        self.unpaywall = unpaywall
        self.max_parallel = max_parallel            # papers looked up at the same time
        self.lookup_timeout_s = lookup_timeout_s    # per source, per paper

    async def find_links(self, paper: PaperRecord) -> list[FullTextLink]:
        pmcid, open_access = await asyncio.gather(self._europepmc(paper), self._unpaywall(paper))
        pmcid = pmcid or open_access.pmcid or _pmcid_in(paper.pdf_url)

        links: list[FullTextLink] = []
        if pmcid and self.pmc_xml:
            links.append(FullTextLink(kind="xml", url=f"{EPMC_REST}/{pmcid}/fullTextXML", via="europepmc"))
        if paper.arxiv_id:
            links.append(FullTextLink(kind="pdf", url=f"https://arxiv.org/pdf/{paper.arxiv_id}", via="arxiv"))
        if paper.pdf_url:
            links.append(FullTextLink(kind="pdf", url=paper.pdf_url, via="metadata"))
        links += [FullTextLink(kind="pdf", url=url, via="unpaywall", license=license)
                  for url, license in open_access.pdfs]
        return _dedupe(links)

    async def find_many(self, papers: dict[int, PaperRecord]) -> dict[int, list[FullTextLink]]:
        """find_links for several papers at once ({key: paper} -> {key: links})."""
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def one(key: int, paper: PaperRecord) -> tuple[int, list[FullTextLink]]:
            async with semaphore:
                return key, await self.find_links(paper)

        return dict(await asyncio.gather(*(one(k, p) for k, p in papers.items())))

    async def _europepmc(self, paper: PaperRecord) -> str | None:
        if self.europepmc is None or not (paper.pmid or paper.doi):
            return None
        try:
            return await asyncio.wait_for(self.europepmc.find_pmcid(pmid=paper.pmid, doi=paper.doi),
                                          timeout=self.lookup_timeout_s)
        except (SourceError, TimeoutError) as e:
            log.warning("full-text lookup skipped (europepmc) for %r: %s", paper.title[:60],
                        e or f"no answer in {self.lookup_timeout_s:.0f}s")
            return None

    async def _unpaywall(self, paper: PaperRecord) -> UnpaywallResult:
        if self.unpaywall is None or not paper.doi:
            return UnpaywallResult()
        try:
            return await asyncio.wait_for(self.unpaywall.lookup(paper.doi), timeout=self.lookup_timeout_s)
        except (SourceError, TimeoutError) as e:
            log.warning("full-text lookup skipped (unpaywall) for %r: %s", paper.title[:60],
                        e or f"no answer in {self.lookup_timeout_s:.0f}s")
            return UnpaywallResult()


def _pmcid_in(url: str | None) -> str | None:
    match = PMCID.search(url or "")
    return match.group(0) if match else None


def _dedupe(links: list[FullTextLink]) -> list[FullTextLink]:
    """Drop repeated URLs (ignoring http/https, case and a trailing slash); keep the first = best."""
    seen: set[str] = set()
    unique = []
    for link in links:
        key = link.url.strip().rstrip("/").lower().replace("http://", "https://")
        if key not in seen:
            seen.add(key)
            unique.append(link)
    return unique
