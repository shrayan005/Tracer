"""Merging duplicate papers (pure logic, no I/O)."""

from app.domain.schemas import PaperRecord
from app.services.dedupe import merge_records, normalize_title


def rec(source, title, query="q", **kw):
    return PaperRecord(source=source, title=title, found_by_query=query, **kw)


def test_normalize_title():
    assert normalize_title("Triceps EMG: A Study!") == "triceps emg a study"


def test_same_doi_merges_and_combines_data():
    [p] = merge_records([
        rec("openalex", "Triceps activation in push-ups", doi="10.1/a", citation_count=10),
        rec("europepmc", "Triceps activation in push-ups.", doi="10.1/a", pmid="99",
            citation_count=12, abstract="Longer abstract text"),
    ])
    assert p.paper.pmid == "99" and p.paper.citation_count == 12 and p.paper.abstract
    assert {f.source for f in p.found_by} == {"openalex", "europepmc"}


def test_chain_of_ids_merges():
    # A has a DOI, B has DOI + PubMed id, C has only the PubMed id: all the same paper
    papers = merge_records([rec("openalex", "Paper one about elbow torque", doi="10.1/x"),
                            rec("europepmc", "Paper one about elbow torque", doi="10.1/x", pmid="5"),
                            rec("semantic_scholar", "Paper 1 elbow torque", pmid="5")])
    assert len(papers) == 1 and papers[0].paper.pmid == "5"


def test_title_match_only_for_long_titles():
    long_title = "Electromyographic analysis of the triceps brachii during exercises"
    assert len(merge_records([rec("a", long_title), rec("b", long_title.upper())])) == 1
    assert len(merge_records([rec("a", "Introduction"), rec("b", "Introduction")])) == 2


def test_same_paper_from_two_queries_records_both():
    [p] = merge_records([rec("openalex", "Elbow extension moment arms of the triceps", "q1", doi="10.9/z"),
                         rec("openalex", "Elbow extension moment arms of the triceps", "q2", doi="10.9/z")])
    assert [f.query for f in p.found_by] == ["q1", "q2"]


def test_retracted_flag_sticks():
    [p] = merge_records([rec("a", "A long enough title about muscles", doi="10.5/r"),
                         rec("b", "A long enough title about muscles", doi="10.5/r", is_retracted=True)])
    assert p.paper.is_retracted
