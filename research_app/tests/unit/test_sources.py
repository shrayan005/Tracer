"""Source parsers tested with small sample responses: no network, no API keys, instant."""

from app.integrations.sources import arxiv, europepmc, openalex, semantic_scholar
from app.integrations.sources.base import last_path_part, normalize_doi, parse_date, strip_html


def test_helpers():
    assert normalize_doi("https://doi.org/10.1000/ABC") == "10.1000/abc"
    assert normalize_doi(None) is None
    assert last_path_part("https://pubmed.ncbi.nlm.nih.gov/2129879") == "2129879"
    assert parse_date("2023-03-15T10:00:00Z").isoformat() == "2023-03-15"
    assert parse_date("not a date") is None
    assert strip_html("EMG <i>activity</i>  increased") == "EMG activity increased"


def test_openalex_record_and_abstract():
    work = {
        "id": "https://openalex.org/W123", "doi": "https://doi.org/10.1/ABC",
        "display_name": "Triceps EMG during push-ups", "publication_year": 2021,
        "publication_date": "2021-05-01", "type": "article", "cited_by_count": 12, "is_retracted": False,
        "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/555"},
        "open_access": {"is_oa": True, "oa_url": "https://x.org/paper"},
        "primary_location": {"source": {"display_name": "J Biomech", "type": "journal"},
                             "pdf_url": "https://x.org/paper.pdf", "landing_page_url": "https://x.org"},
        "authorships": [{"author": {"display_name": "Ram Sharma"}}],
        "abstract_inverted_index": {"Triceps": [0], "activation": [1, 3], "increased": [2]},
    }
    r = openalex.to_record(work, "triceps EMG")
    assert (r.doi, r.openalex_id, r.pmid) == ("10.1/abc", "W123", "555")
    assert r.abstract == "Triceps activation increased activation"
    assert r.venue == "J Biomech" and r.authors == ["Ram Sharma"] and r.pdf_url.endswith(".pdf")
    assert openalex.to_record({"display_name": None}, "q") is None       # no title -> skipped


def test_europepmc_record():
    item = {"id": "123", "source": "MED", "pmid": "123", "doi": "10.2/XYZ", "title": "Bench press EMG.",
            "authorString": "Lee A, Kim B.", "pubYear": "2024", "firstPublicationDate": "2024-02-01",
            "citedByCount": 4, "isOpenAccess": "Y", "abstractText": "We measured <b>EMG</b>.",
            "pubTypeList": {"pubType": ["Review", "Journal Article"]},
            "journalInfo": {"journal": {"title": "Sports Med"}},
            "fullTextUrlList": {"fullTextUrl": [
                {"documentStyle": "pdf", "availability": "Open access", "url": "https://e.org/a.pdf"}]}}
    r = europepmc.to_record(item, "q")
    assert r.title == "Bench press EMG" and r.doi == "10.2/xyz" and r.year == 2024
    assert r.authors == ["Lee A", "Kim B"] and r.publication_type == "review"
    assert r.abstract == "We measured EMG ." and r.is_open_access is True and r.pdf_url == "https://e.org/a.pdf"


ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2209.15001v3</id>
    <published>2022-09-29T17:57:08Z</published>
    <title>Dilated Neighborhood
      Attention Transformer</title>
    <summary>  We present DiNAT. </summary>
    <author><name>Ali Hassani</name></author>
    <arxiv:doi>10.48550/ARXIV.2209.15001</arxiv:doi>
    <link title="pdf" href="https://arxiv.org/pdf/2209.15001v3"/>
  </entry>
</feed>"""


def test_arxiv_feed():
    [r] = arxiv.parse_feed(ARXIV_FEED, "attention")
    assert r.arxiv_id == "2209.15001" and r.title == "Dilated Neighborhood Attention Transformer"
    assert r.year == 2022 and r.abstract == "We present DiNAT." and r.publication_type == "preprint"
    assert r.doi == "10.48550/arxiv.2209.15001" and r.pdf_url.endswith("v3")


def test_semantic_scholar_record():
    paper = {"paperId": "abc", "title": "A review of triceps", "year": 2020,
             "externalIds": {"DOI": "10.3/Q", "PubMed": 77}, "publicationTypes": ["Review"],
             "citationCount": 50, "influentialCitationCount": 5, "tldr": {"text": "Short."},
             "authors": [{"name": "X"}], "openAccessPdf": {"url": ""}}
    r = semantic_scholar.to_record(paper, "q")
    assert (r.s2_id, r.doi, r.pmid) == ("abc", "10.3/q", "77")
    assert r.publication_type == "review" and r.tldr == "Short." and r.pdf_url is None
