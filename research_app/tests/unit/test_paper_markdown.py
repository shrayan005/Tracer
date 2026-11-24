"""Single-paper detail page (step 5.7): pure rendering of a PaperDetail."""

from app.domain.schemas import DocumentInfo, PaperDetail, SectionInfo, TrustAssessment
from app.presentation.paper_markdown import render_paper
from app.services.quality import score
from tests.unit.test_report_markdown import report_paper
from tests.unit.test_report_milestone5 import analysed, m5_report


def detail(paper_rank=2, document=True, connections=True) -> PaperDetail:
    report = m5_report()
    paper = next(p for p in report.papers if p.rank == paper_rank)
    doc = DocumentInfo(content_type="application/pdf", source_url="https://pub.org/a.pdf", size_bytes=700_000,
                       page_count=10, status="extracted",
                       sections=[SectionInfo(kind="methods", title="Methods", characters=13_100),
                                 SectionInfo(kind="results", title="Results", characters=6_000),
                                 SectionInfo(kind="introduction", title="Intro", characters=3_800)])
    return PaperDetail(topic_id=12, topic="biomechanics of triceps", field_family="sport_biomechanics",
                       n_papers=len(report.papers), paper=paper,
                       connections=report.connections if connections else [],
                       labels={p.paper_id: p.in_text for p in report.papers},
                       ranks={p.paper_id: p.rank for p in report.papers}, document=doc if document else None)


def test_all_parts_in_order():
    md = render_paper(detail())
    headings = ["# Fascicle training", "## Claims", "## Quality appraisal", "## Trust: moderate → low",
                "## Place in this topic", "## Full text used", "## Cite this paper"]
    positions = [md.index(h) for h in headings]
    assert positions == sorted(positions)
    assert '**In topic #12 "biomechanics of triceps":** rank 2 of 2 · relevance 9/10 (on topic)' in md


def test_claims_table_with_quotes_and_escaped_pipes():
    d = detail()
    d.paper.analysis.analysis.claims[0].quote = "Strength | rose 40%."
    md = render_paper(d)
    assert ('| 1 | ↑ training (before vs after) → 1-RM (+40%) | 9 women | "Strength \\| rose 40%." |') in md
    assert "> ⚠ claim 2: quote not found in the paper" in md


def test_appraisal_table_shows_evidence_and_matches_the_quality_score():
    d = detail()
    d.paper.analysis.analysis.appraisal.randomized.quote = "Arms were randomly assigned."
    md = render_paper(d)
    assert '| Randomised | ✓ yes | "Arms were randomly assigned." |' in md
    assert "| Statistics | ? not reported |  |" in md                         # "?" rows have no quote
    assert "| Blinded |" not in md                                          # not applicable: left out
    assert "| Effect sizes | ? not reported | ⚠ the AI said yes but could not quote it" in md
    earned, possible, _ = score(d.paper.analysis.analysis.appraisal, "sport_biomechanics")
    assert f"**Score:** {earned:g} of {possible:g} applicable points ({earned / possible:.0%})" in md


def test_trust_chain_changed_and_unchanged():
    md = render_paper(detail())
    assert "1. **From the study design:** moderate (3/5): lab experiment; 33 citations" in md
    assert "2. **From the quality check:** quality as expected for its design" in md
    assert "3. **Red flag:** very small sample (n=9): trust -1" in md
    assert "**Result: low (2/5)**" in md

    unchanged = detail(paper_rank=1)                                         # no analysis, no base trust
    md = render_paper(unchanged)
    assert "## Trust: low" in md and "2. **From the quality check:** not analysed yet" in md


def test_place_in_topic_puts_this_paper_first():
    md = render_paper(detail())
    agree = md.split("### ✓ Agree: Does training increase strength?")[1].split("###")[0]
    lines = [l for l in agree.splitlines() if l.startswith("- ")]
    assert lines[0].startswith("- ↑ **this paper** supports: 1-RM (+40%)")
    assert "[(Smith & Lee, 2021)](12-biomechanics-of-triceps.md#1)" in lines[1]
    assert md.count("**Why they may differ:**") == 1

    alone = render_paper(detail(connections=False))
    assert "_No other selected paper answers the same questions as this one._" in alone


def test_full_text_and_abstract_only():
    md = render_paper(detail())
    assert "PDF from https://pub.org/a.pdf · 683 KB · 10 pages" in md
    assert "Extracted into 3 sections: introduction 3.8k · methods 13.1k · results 6.0k characters" in md
    md = render_paper(detail(document=False))
    assert "_Summarised from the abstract only: no free full text was found._" in md
    assert "Extracted into" not in md


def test_failed_document_explains_why():
    d = detail()
    d.document = DocumentInfo(content_type="application/pdf", source_url=None, size_bytes=1024, page_count=1,
                              status="failed", error="no text layer (scanned PDF): would need OCR")
    md = render_paper(d)
    assert "_Not usable: no text layer (scanned PDF): would need OCR. Summarised from the abstract instead._" in md


def test_paper_without_analysis_still_renders():
    p = report_paper(1, "Old paper", None, "k")
    d = PaperDetail(topic_id=5, topic="t", field_family="general", n_papers=1, paper=p)
    md = render_paper(d)
    assert "_Not analysed yet: run `python -m app.cli analyze <topic id>`._" in md
    assert "## Cite this paper" in md and "@article{k," in md
