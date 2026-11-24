"""Full-text finder + Unpaywall parsing, with fake clients (no network)."""

import asyncio

from app.domain.errors import SourceError
from app.domain.schemas import FullTextLink, PaperRecord
from app.services.fulltext_finder import FullTextFinder
from app.integrations.sources.unpaywall import UnpaywallResult, parse

XML = "https://www.ebi.ac.uk/europepmc/webservices/rest/{}/fullTextXML"


class FakeEuropePmc:
    def __init__(self, pmcid=None, error=None, delay=0.0):
        self.pmcid, self.error, self.delay, self.calls = pmcid, error, delay, 0

    async def find_pmcid(self, *, pmid=None, doi=None):
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.pmcid


class FakeUnpaywall:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result or UnpaywallResult(), error, 0

    async def lookup(self, doi):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def paper(**kwargs) -> PaperRecord:
    return PaperRecord(source="x", title="Triceps study", **kwargs)


# ---------- Unpaywall JSON parsing ----------

def test_parse_collects_pdfs_and_spots_the_pmcid():
    data = {"is_oa": True,
            "best_oa_location": {"url_for_pdf": "https://pub.org/a.pdf", "url": "https://pub.org/a",
                                 "license": "cc-by"},
            "oa_locations": [{"url_for_pdf": "https://pub.org/a.pdf"},                       # duplicate
                             {"url_for_pdf": None, "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8174684/"},
                             {"url_for_pdf": "https://repo.uni.edu/a.pdf", "license": None}]}
    result = parse(data)
    assert result.pdfs == [("https://pub.org/a.pdf", "cc-by"), ("https://repo.uni.edu/a.pdf", None)]
    assert result.pmcid == "PMC8174684"


def test_parse_not_open_access():
    assert parse({"is_oa": False, "best_oa_location": {"url_for_pdf": "x"}}) == UnpaywallResult()


# ---------- the finder ----------

async def test_links_are_ordered_best_first_and_deduplicated():
    finder = FullTextFinder(FakeEuropePmc(pmcid="PMC1234567"), FakeUnpaywall(UnpaywallResult(
        pdfs=[("https://pub.org/a.pdf", "cc-by"), ("http://arxiv.org/pdf/2401.00001/", None)])))
    links = await finder.find_links(paper(doi="10.1/a", pmid="1", arxiv_id="2401.00001",
                                          pdf_url="https://pub.org/a.pdf"))
    assert [(l.kind, l.via) for l in links] == [("xml", "europepmc"), ("pdf", "arxiv"), ("pdf", "metadata")]
    assert links[0].url == XML.format("PMC1234567")


async def test_pmcid_from_unpaywall_when_europepmc_is_down():
    finder = FullTextFinder(FakeEuropePmc(error=SourceError("europepmc", "network error")),
                            FakeUnpaywall(UnpaywallResult(pmcid="PMC7654321")))
    links = await finder.find_links(paper(doi="10.1/a", pmid="1"))
    assert links == [FullTextLink(kind="xml", url=XML.format("PMC7654321"), via="europepmc")]


async def test_slow_source_is_cut_off_by_the_time_limit():
    finder = FullTextFinder(FakeEuropePmc(pmcid="PMC1111111", delay=5), FakeUnpaywall(UnpaywallResult(
        pdfs=[("https://pub.org/a.pdf", None)])), lookup_timeout_s=0.05)
    links = await finder.find_links(paper(doi="10.1/a"))
    assert [l.via for l in links] == ["unpaywall"]                 # europepmc skipped, no crash


async def test_paper_without_ids_makes_no_requests():
    europepmc, unpaywall = FakeEuropePmc(pmcid="PMC1"), FakeUnpaywall()
    assert await FullTextFinder(europepmc, unpaywall).find_links(paper()) == []
    assert europepmc.calls == 0 and unpaywall.calls == 0


async def test_find_many_keeps_keys_and_works_without_clients():
    finder = FullTextFinder(None, None)
    found = await finder.find_many({3: paper(arxiv_id="2401.00001"), 8: paper()})
    assert [l.via for l in found[3]] == ["arxiv"] and found[8] == []


async def test_pmc_xml_switched_off_gives_pdfs_only():
    """EUROPEPMC_FULLTEXT=false: a PMC id from Unpaywall no longer becomes a Europe PMC XML link."""
    finder = FullTextFinder(None, FakeUnpaywall(UnpaywallResult(
        pmcid="PMC7654321", pdfs=[("https://repo.org/a.pdf", "cc-by")])), pmc_xml=False)
    links = await finder.find_links(paper(doi="10.1/x"))
    assert [link.via for link in links] == ["unpaywall"]
