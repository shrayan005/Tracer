from datetime import datetime, timezone

from app.domain.schemas import (PaperRecord, PaperSummary, ReportPaper, ResearchPlan, ResearchReport,
                                SummarizedPaper, TrustAssessment)
from app.presentation.report_markdown import render, slugify
from app.services.citations import apa, bibtex, in_text

PLAN = ResearchPlan(field="sports science", is_biomedical=True, is_cs_math_physics=False,
                    research_brief="How the triceps extends the elbow.", queries=["triceps EMG", "elbow torque"])


def report_paper(rank: int, title: str, summary: SummarizedPaper | None, key: str) -> ReportPaper:
    paper = PaperRecord(source="db", title=title, authors=["John Smith", "Kim Lee"], year=2021,
                        venue="J Biomech", doi=f"10.1/{rank}", citation_count=12)
    return ReportPaper(rank=rank, paper_id=rank, paper=paper, relevance=9, relevance_reason="on topic",
                       study_type="lab_experiment",
                       trust=TrustAssessment(level=2, label="low", reasons=["lab experiment", "12 citations"]),
                       score=0.71, score_parts={"relevance": 0.45, "trust": 0.1}, summary=summary,
                       apa=apa(paper), in_text=in_text(paper), bibtex_key=key, bibtex=bibtex(paper, key))


def make_report(papers=None, warnings=None) -> ResearchReport:
    full = SummarizedPaper(based_on="abstract", flags=["number not found in abstract: 42"], summary=PaperSummary(
        tldr="Grip width changes triceps activation.", objective="Compare grips.", methods="24 men, EMG.",
        key_findings=["Narrow grip raised activation by 42%"], why_relevant="Directly about triceps."))
    title_only = SummarizedPaper(based_on="title only", summary=None,
                                 flags=["no abstract: needs the full text (milestone 4)"])
    if papers is None:
        papers = [report_paper(1, "Triceps | grip width", full, "smith2021tricepsa"),
                  report_paper(2, "Triceps tendon anatomy", title_only, "smith2021tricepsb"),
                  report_paper(3, "Elbow torque model", None, "smith2021elbow")]
    return ResearchReport(topic_id=12, topic="biomechanics of triceps",
                          created_at=datetime(2026, 11, 20, tzinfo=timezone.utc), plan=PLAN,
                          field_family="biomedical", sources_used=["openalex", "europepmc"],
                          counts={"found": 146, "prefiltered": 40, "screened": 40, "selected": len(papers)},
                          timings_s={"discovery": 30.2, "screening": 22.1, "summaries": 11.9},
                          warnings=warnings or [], papers=papers)


def test_report_has_every_section():
    md = render(make_report(warnings=["europepmc timed out"]))
    for heading in ("# Research report: biomechanics of triceps", "## Overview", "## Papers",
                    "## References (APA 7)", "## BibTeX", "```bibtex"):
        assert heading in md
    assert "> How the triceps extends the elbow." in md
    assert "146 papers found → 40 pre-filtered → 40 screened by the LLM → **3 selected** · 64s" in md
    assert "- europepmc timed out" in md


def test_summary_variants_and_flags():
    md = render(make_report())
    assert "**TL;DR:** Grip width changes triceps activation." in md
    assert "> ⚠ Check against the paper: number not found in abstract: 42" in md
    assert "**Limitations (stated by the authors):** none stated in the abstract" in md
    assert "_No abstract and no free full text: only the title is known._" in md
    assert "_Summary unavailable: the LLM call failed." in md


def test_table_cells_are_escaped_and_bibtex_keys_present():
    md = render(make_report())
    assert "Triceps \\| grip width" in md                  # a '|' in a title must not break the table
    assert "@article{smith2021tricepsa," in md and "@article{smith2021tricepsb," in md
    assert "| **Trust** | low (2/5): lab experiment; 12 citations |" in md


def test_score_parts_in_fixed_order_and_no_duplicate_pdf_link():
    paper = report_paper(1, "T", None, "k")
    paper.score_parts = {"trust": 0.1, "full_text": 0.05, "relevance": 0.45}     # JSONB-shuffled order
    paper.paper.pdf_url = "https://doi.org/10.1/1"                                # same as the DOI link
    md = render(make_report(papers=[paper]))
    assert "| **Score** | 0.71 = relevance 0.45 + trust 0.10 + full text 0.05 |" in md
    assert "[PDF]" not in md


def test_empty_selection_says_so():
    md = render(make_report(papers=[]))
    assert "No papers passed the selection" in md
    assert "## Overview" not in md


def test_slugify():
    assert slugify("Biomechanics of Triceps!") == "biomechanics-of-triceps"
    assert slugify("???") == "topic"
    assert len(slugify("x" * 200)) == 50


def test_quotes_and_full_text_count_are_shown():
    finding = SummarizedPaper(based_on="full text", summary=PaperSummary(
        tldr="t", objective="o", methods="m", why_relevant="w", limitations=[],
        key_findings=[{"statement": "Heads do not work in unison.",
                       "quote": "the three heads do not work in unison under any of the conditions"}]))
    report = make_report(papers=[report_paper(1, "Fatigue of the triceps", finding, "k")])
    report.counts["full_text"] = 1
    md = render(report)
    assert '- Heads do not work in unison.\n  > "the three heads do not work in unison under any of the conditions"' in md
    assert "**1 selected** · full text for 1 of 1 ·" in md
    assert "**Limitations (stated by the authors):** none stated in the paper" in md
    assert "| **Summary based on** | full text |" in md
