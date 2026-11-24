"""Milestone 5 in the Markdown report (step 5.6): connections, gaps, claims, checklist, trust change."""

from app.domain.schemas import (AnalyzedPaper, Connection, ConnectionMember, EvidenceGaps, GapWriteUp,
                                ResearchReport, SharedLimitation, TrustAssessment)
from app.presentation.report_markdown import render
from tests.unit.test_report_markdown import make_report, report_paper

FACTS = ("control_group", "randomized", "blinded", "statistics_reported", "effect_sizes",
         "conflicts_declared", "data_available")


def analysed(**facts) -> AnalyzedPaper:
    appraisal = {"sample_size": 9, "sex": "female", "duration": "under 3 months",
                 **{f: {"value": facts.get(f, "not reported")} for f in FACTS}}
    return AnalyzedPaper(based_on="full text", flags=["claim 2: quote not found in the paper",
                                                      "effect_sizes: 'yes' not supported by a quote -> not reported"],
                         analysis={"appraisal": appraisal, "claims": [
                             {"factor": "training (before vs after)", "outcome": "1-RM", "direction": "increase",
                              "magnitude": "+40%", "population": "9 women", "quote": "1-RM increased 40%."},
                             {"factor": "training (before vs after)", "outcome": "muscle thickness",
                              "direction": "no difference", "population": "9 women", "quote": "No change."}]})


def m5_report() -> ResearchReport:
    p1 = report_paper(1, "Training and strength", None, "a")
    p2 = report_paper(2, "Fascicle training", None, "b")
    p2.base_trust = TrustAssessment(level=3, label="moderate", reasons=["lab experiment", "33 citations"])
    p2.trust = TrustAssessment(level=2, label="low", reasons=[
        "lab experiment", "33 citations", "quality 3/6.5: control group ✓, n=9 (small)",
        "quality as expected for its design; very small sample (n=9): trust -1"])
    p2.analysis = analysed(control_group="yes", randomized="yes", blinded="not applicable", data_available="no")
    claims = p2.analysis.analysis.claims
    report = make_report(papers=[p1, p2])
    report.connections = [
        Connection(question="Does training increase strength?", relation="agree", finding="Both report gains.",
                   members=[ConnectionMember(paper=1, claim=0, stance="supports", claim_snapshot=claims[0]),
                            ConnectionMember(paper=2, claim=0, stance="supports", claim_snapshot=claims[0])]),
        Connection(question="Does training increase muscle thickness?", relation="mixed", finding="Mixed.",
                   why_differ="Different durations.",
                   members=[ConnectionMember(paper=1, claim=1, stance="no effect", claim_snapshot=claims[1]),
                            ConnectionMember(paper=2, claim=0, stance="supports", claim_snapshot=claims[0])])]
    report.gaps = EvidenceGaps(
        n_papers=2, gaps=["No study reports blinding."], papers_with_limitations=1,
        shared_limitations=[SharedLimitation(theme="small sample", papers=["(A, 2020)", "(B, 2021)"])],
        write_up=GapWriteUp(summary="The evidence is thin.", next_studies=["A larger trial."]),
        flags=["suggested study dropped (uses numbers not in the counted facts: 40): x"])
    report.synthesis_flags = ["group 'X?' dropped: fewer than 2 papers left"]
    return report


def test_synthesis_sections_come_first_in_order():
    md = render(m5_report())
    order = [md.index(h) for h in ("## Overview", "## How the papers connect", "## What the evidence is missing",
                                   "## Papers", "## References (APA 7)")]
    assert order == sorted(order)


def test_connections_section():
    md = render(m5_report())
    assert "2 questions answered by more than one paper: ✓ 1 agree · ≈ 1 mixed" in md
    assert "### ✓ Agree: Does training increase strength?" in md
    assert "### ≈ Mixed: Does training increase muscle thickness?" in md
    assert ("- ↑ **supports** · [(Smith & Lee, 2021)](#2): training (before vs after) → 1-RM (+40%), 9 women\n"
            '  > "1-RM increased 40%."') in md
    assert "- = **no effect**" in md
    assert md.count("**Why they may differ:**") == 1                       # only for the disagreement


def test_gaps_section():
    md = render(m5_report())
    assert "_Counted by code across the 2 papers:_\n\n- No study reports blinding." in md
    assert "**Limitations the authors admit** (1 of 2 papers state any):" in md
    assert "- small sample: (A, 2020), (B, 2021)" in md
    assert "The evidence is thin." in md and "- A larger trial." in md


def test_paper_section_has_claims_checklist_and_trust_change():
    md = render(m5_report())
    assert "**Claims (structured)**\n\n- ↑ training (before vs after) → 1-RM (+40%) · 9 women" in md
    assert ("**Quality checklist:** n = 9 · female · under 3 months · ✓ control group · ✓ randomised · ? statistics"
            " · ? effect sizes · ? conflicts stated · ✗ data shared") in md
    checklist_line = md.split("**Quality checklist:**")[1].split("\n")[0]
    assert "blinded" not in checklist_line                                       # not applicable: left out
    assert "> ⚠ claim 2: quote not found in the paper" in md                     # reader-relevant: shown
    assert "| moderate → low | 9/10 |" in md                                     # Overview
    assert ("| **Trust** | moderate → low (2/5): lab experiment; 33 citations; quality as expected for its design; "
            "very small sample (n=9): trust -1 |") in md
    assert "quality 3/6.5" not in md                                             # not repeated in the Trust row


def test_internal_checks_are_folded_away():
    md = render(m5_report())
    box = md.split("<details><summary>Checks applied (3)</summary>")[1]
    assert "group 'X?' dropped" in box and "suggested study dropped" in box
    assert "(Smith & Lee, 2021): effect_sizes: 'yes' not supported" in box
    main_text = md.split("<details>")[0]
    assert "group 'X?' dropped" not in main_text and "effect_sizes: 'yes'" not in main_text


def test_level_change_within_the_same_label_shows_the_levels():
    report = m5_report()
    p1 = report.papers[0]
    p1.base_trust = TrustAssessment(level=4, label="high", reasons=["randomized trial"])
    p1.trust = TrustAssessment(level=5, label="high", reasons=["randomized trial", "quality 6/6.5: ...",
                                                              "well-conducted study: trust +1"])
    md = render(report)
    assert "| high (4 → 5) | 9/10 |" in md and "high → high" not in md


def test_notes_are_shown_apart_from_the_gaps():
    report = m5_report()
    report.gaps.notes = ["6 of 10 papers could only be judged from their abstracts."]
    md = render(report)
    assert "_Note: 6 of 10 papers could only be judged from their abstracts._" in md


def test_report_without_milestone_5_has_none_of_the_new_parts():
    md = render(make_report())
    for part in ("How the papers connect", "What the evidence is missing", "Claims (structured)",
                 "Quality checklist", "<details>", "→ low"):
        assert part not in md
