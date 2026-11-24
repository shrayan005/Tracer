"""
Markdown report (presentation layer): turns a ResearchReport into text a human reads.

A pure function: no database, no LLM, no files. The CLI decides where the text goes (a file);
later the web UI can render the same ResearchReport differently. That's why formatting lives
here and not in the service.
"""

import re

from app.domain.schemas import ReportPaper, ResearchReport
from app.services.selection import WEIGHTS

PART_ORDER = list(WEIGHTS)        # relevance, trust, citations, recency, full_text
QUALITY_DETAIL = re.compile(r"^quality [\d.]+/[\d.]+:")      # "quality 3/6.5: control group ✓, ..." (5.2)

STUDY_NAMES = {"meta_analysis": "Meta-analysis", "systematic_review": "Systematic review",
               "narrative_review": "Narrative review", "randomized_trial": "Randomized trial",
               "cohort": "Cohort", "case_control": "Case-control", "cross_sectional": "Cross-sectional",
               "case_report": "Case report", "lab_experiment": "Lab experiment",
               "computational": "Computational", "theoretical": "Theoretical", "other": "Other"}


def _cell(text: str) -> str:
    """Make text safe inside a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def slugify(text: str, max_length: int = 50) -> str:
    """'Biomechanics of Triceps!' -> 'biomechanics-of-triceps' (for file names)."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_length].rstrip("-") or "topic"


def render(report: ResearchReport) -> str:
    out: list[str] = []
    add = out.append
    plan, counts = report.plan, report.counts
    date = f"{report.created_at:%Y-%m-%d}" if report.created_at else "unknown date"

    add(f"# Research report: {report.topic}")
    add("")
    add(f"> {plan.research_brief}")
    add("")
    add(f"- **Field:** {plan.field} (evidence rules: {report.field_family}) · **Date:** {date} · "
        f"**Topic id:** {report.topic_id}")
    add(f"- **Sources:** {', '.join(report.sources_used) or 'none'}")
    add("- **Search queries:** " + "; ".join(f"`{q}`" for q in plan.queries))
    skipped = (f" ({counts['skipped']} more not needed: enough relevant papers already)"
               if counts.get("skipped") else "")
    add(f"- **Pipeline:** {counts.get('found', 0)} papers found → {counts.get('prefiltered', 0)} pre-filtered → "
        f"{counts.get('screened', 0)} screened by the LLM{skipped} → "
        + (f"{counts['shortlisted']} shortlisted → " if "shortlisted" in counts else "")
        + f"**{counts.get('selected', 0)} selected** "
        + (f"· full text for {counts['full_text']} of {counts.get('selected', 0)} " if "full_text" in counts else "")
        + f"· {sum(report.timings_s.values()):.0f}s "
        f"({', '.join(f'{stage} {s:.0f}s' for stage, s in report.timings_s.items())})")
    if report.warnings:
        add("")
        add("**⚠ Warnings**")
        add("")
        for warning in report.warnings:
            add(f"- {warning}")
    add("")

    if not report.papers:
        add("_No papers passed the selection. Try a broader or differently worded topic._")
        return "\n".join(out) + "\n"

    add("## Overview")
    add("")
    add("| # | Paper | Year | Design | Trust | Relevance | Score |")
    add("|---|---|---|---|---|---|---|")
    for p in report.papers:
        add(f"| {p.rank} | [{_cell(p.paper.title)}](#{p.rank}) {_cell(p.in_text)} | {p.paper.year or '?'} | "
            f"{STUDY_NAMES.get(p.study_type, p.study_type)} | {_trust_text(p)} | {p.relevance}/10 | {p.score:.2f} |")
    add("")

    out.extend(_connections_section(report))       # the answer first: what the papers say TOGETHER
    out.extend(_gaps_section(report))

    add("## Papers")
    add("")
    for p in report.papers:
        out.extend(_paper_section(p))

    add("## References (APA 7)")
    add("")
    for p in sorted(report.papers, key=lambda p: p.apa.lower()):
        add(f"- {p.apa}")
    add("")
    add("## BibTeX")
    add("")
    add("```bibtex")
    add("\n\n".join(p.bibtex for p in report.papers))
    add("```")
    add("")
    add("---")
    add("_Summaries use the free full text where one exists (key sections, shortened) and the abstract "
        "otherwise; every quote and number was checked against the paper text, and anything not found is "
        "flagged with ⚠. A quote proves the words exist in the paper, not that the summary reads them "
        "correctly. Trust starts from the study design, peer review and citations, and moves at most one step "
        "for how well the study was done (its checklist; only judged from the full text) or for a very small "
        "sample. Agree/contradict labels are computed by code from each paper's stated result, and the gaps "
        "are counted by code, not written by the AI. Results can vary slightly between runs. "
        "Always check the original paper before relying on a finding._")
    out.extend(_checks_box(report))
    return "\n".join(out) + "\n"


# ---------- milestone 5: connections, gaps, claims, checklist ----------

ARROWS = {"increase": "↑", "decrease": "↓", "no difference": "=", "association": "~", "other": "·"}
RELATIONS = {"agree": ("✓", "Agree"), "contradict": ("✗", "Contradict"), "mixed": ("≈", "Mixed")}
CHECKLIST = {"control_group": "control group", "randomized": "randomised", "blinded": "blinded",
             "statistics_reported": "statistics", "effect_sizes": "effect sizes",
             "conflicts_declared": "conflicts stated", "data_available": "data shared"}
MARKS = {"yes": "✓", "no": "✗", "not reported": "?"}


def _trust_text(p: ReportPaper) -> str:
    """'moderate', 'moderate → low' when the quality check (5.2) changed it, or 'high (4 → 5)' when the
    level changed but the label didn't (levels 4 and 5 are both "high")."""
    base = p.base_trust or p.trust
    if base.level == p.trust.level:
        return p.trust.label
    if base.label == p.trust.label:
        return f"{p.trust.label} ({base.level} → {p.trust.level})"
    return f"{base.label} → {p.trust.label}"


def _connections_section(report: ResearchReport) -> list[str]:
    if not report.connections:
        return []
    rank = {p.paper_id: p.rank for p in report.papers}
    label = {p.paper_id: p.in_text for p in report.papers}
    counts = {r: sum(c.relation == r for c in report.connections) for r in RELATIONS}
    out = ["## How the papers connect", "",
           f"{len(report.connections)} questions answered by more than one paper: "
           + " · ".join(f"{RELATIONS[r][0]} {n} {r}" for r, n in counts.items() if n), ""]
    for c in report.connections:
        icon, word = RELATIONS[c.relation]
        out += [f"### {icon} {word}: {c.question}", "", c.finding, ""]
        for m in c.members:
            claim = m.claim_snapshot
            if claim is None:
                continue
            magnitude = f" ({claim.magnitude})" if claim.magnitude else ""
            out.append(f"- {ARROWS[claim.direction]} **{m.stance}** · [{label.get(m.paper, '?')}](#{rank.get(m.paper, '')}): "
                       f"{claim.factor} → {claim.outcome}{magnitude}, {claim.population}")
            out.append(f'  > "{claim.quote}"')
        if c.why_differ:
            out += ["", f"**Why they may differ:** {c.why_differ}"]
        out.append("")
    return out


def _gaps_section(report: ResearchReport) -> list[str]:
    g = report.gaps
    if g is None:
        return []
    out = ["## What the evidence is missing", "", f"_Counted by code across the {g.n_papers} papers:_", ""]
    out += [f"- {gap}" for gap in g.gaps] or ["- No gaps found by the rules."]
    out += [""] + [f"_Note: {note}_" for note in g.notes] if g.notes else []
    out += ["", f"**Limitations the authors admit** ({g.papers_with_limitations} of {g.n_papers} papers state any):", ""]
    out += [f"- {s.theme}: {', '.join(s.papers)}" for s in g.shared_limitations] \
        or ["- none shared by two or more papers"]
    if g.write_up:
        out += ["", g.write_up.summary]
        if g.write_up.next_studies:
            out += ["", "**Studies that would fill these gaps:**", ""] + [f"- {s}" for s in g.write_up.next_studies]
    return out + [""]


def _paper_analysis_lines(p: ReportPaper) -> list[str]:
    """Inside a paper's section: its structured claims and quality checklist (milestone 5)."""
    if p.analysis is None:
        return []
    a = p.analysis.analysis
    out = ["**Claims (structured)**", ""]
    for c in a.claims:
        magnitude = f" ({c.magnitude})" if c.magnitude else ""
        out.append(f"- {ARROWS[c.direction]} {c.factor} → {c.outcome}{magnitude} · {c.population}")
    ap = a.appraisal
    n = f"n = {ap.sample_size}" if ap.sample_size is not None else "n = ?"
    items = " · ".join(f"{MARKS[getattr(ap, k).value]} {name}" for k, name in CHECKLIST.items()
                       if getattr(ap, k).value != "not applicable")
    out += ["", f"**Quality checklist:** {n} · {ap.sex} · {ap.duration} · {items} "
                f"_(✓ shown with a quote · ✗ stated not done · ? not reported)_", ""]
    unverified = [f for f in p.analysis.flags if f.startswith("claim ")]      # reader-relevant: shown here
    out += [f"> ⚠ {flag}" for flag in unverified]
    return out + ([""] if unverified else [])


def _checks_box(report: ResearchReport) -> list[str]:
    """Internal check messages: transparent, but folded away so they don't clutter the report."""
    checks = list(report.synthesis_flags) + (list(report.gaps.flags) if report.gaps else [])
    for p in report.papers:
        if p.analysis is not None:
            checks += [f"{p.in_text}: {f}" for f in p.analysis.flags if not f.startswith("claim ")]
    if not checks:
        return []
    return ["", f"<details><summary>Checks applied ({len(checks)})</summary>", ""] + \
        [f"- {c}" for c in checks] + ["", "</details>"]


def _paper_section(p: ReportPaper) -> list[str]:
    lines = [f'<a id="{p.rank}"></a>', f"### {p.rank}. {p.paper.title}", "", p.apa, ""]
    s = p.summary
    if s is None:
        lines += ["_Summary unavailable: the LLM call failed. Run the topic again later._", ""]
    elif s.summary is None:
        lines += ["_No abstract and no free full text: only the title is known._", ""]
    else:
        body = s.summary
        lines += [f"**TL;DR:** {body.tldr}", "",
                  f"- **Objective:** {body.objective}",
                  f"- **Methods:** {body.methods}", "",
                  "**Key findings**", ""]
        for finding in body.key_findings:
            lines.append(f"- {finding.statement}")
            if finding.quote:                                   # the evidence, copied from the paper
                lines.append(f'  > "{finding.quote}"')
        lines.append("")
        if s.flags:
            lines += [f"> ⚠ Check against the paper: {flag}" for flag in s.flags]
            lines.append("")
        where = "paper" if s.based_on == "full text" else "abstract"
        limitations = "; ".join(body.limitations) if body.limitations else f"none stated in the {where}"
        lines += [f"**Limitations (stated by the authors):** {limitations}", "",
                  f"**Why it's relevant:** {body.why_relevant}", ""]
    lines += _paper_analysis_lines(p)

    # fixed order: the database (JSONB) does not keep the order the keys were saved in
    ordered = sorted(p.score_parts.items(), key=lambda kv: PART_ORDER.index(kv[0]) if kv[0] in PART_ORDER else 99)
    parts = " + ".join(f"{name.replace('_', ' ')} {value:.2f}" for name, value in ordered)
    links = []
    doi_url = f"https://doi.org/{p.paper.doi}" if p.paper.doi else None
    if doi_url:
        links.append(f"[DOI]({doi_url})")
    if p.paper.pdf_url and p.paper.pdf_url.lower() != (doi_url or "").lower():   # some sources give the DOI as "PDF"
        links.append(f"[PDF]({p.paper.pdf_url})")
    if p.paper.landing_url and not p.paper.doi:
        links.append(f"[Page]({p.paper.landing_url})")
    based_on = s.based_on if s else "nothing (failed)"
    lines += [
        "| | |", "|---|---|",
        f"| **Design** | {STUDY_NAMES.get(p.study_type, p.study_type)} |",
        # the "quality 3/6.5: ..." detail is left out here (the checklist line above shows it);
        # the conclusion ("quality as expected ...; very small sample ...") stays
        f"| **Trust** | {_trust_text(p)} ({p.trust.level}/5): "
        f"{_cell('; '.join(r for r in p.trust.reasons if not QUALITY_DETAIL.match(r)))} |",
        f"| **Relevance** | {p.relevance}/10: {_cell(p.relevance_reason)} |",
        f"| **Score** | {p.score:.2f} = {parts} |",
        f"| **Venue** | {_cell(p.paper.venue or '-')} · cited {p.paper.citation_count or 0} times |",
        f"| **Links** | {' · '.join(links) or '-'} |",
        f"| **Summary based on** | {based_on} |",
        "",
    ]
    return lines
