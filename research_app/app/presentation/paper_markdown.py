"""
Single-paper detail view (presentation layer, step 5.7): everything known about ONE paper of a topic.

A pure function like report_markdown.render(): no database, no LLM. It shows the evidence behind every
judgement: a quote next to every claim and every ✓, the trust level as a chain of steps, the paper's
place among the other papers, and which text the AI actually read.
The symbols and helpers are shared with the topic report, so both pages always mean the same thing.
"""

from app.domain.schemas import PaperDetail, ReportPaper
from app.presentation.report_markdown import (ARROWS, CHECKLIST, MARKS, PART_ORDER, RELATIONS, STUDY_NAMES,
                                              _cell, _trust_text, slugify)
from app.services.quality import score as quality_score

KIND_ORDER = ["abstract", "introduction", "methods", "results", "table", "discussion", "limitations",
              "conclusion", "caption", "backmatter", "references", "front", "other"]
FACT_FLAG = "{name}: 'yes' not supported by a quote"


def render_paper(d: PaperDetail) -> str:
    p = d.paper
    out = _header(d) + _summary(p) + _claims(p) + _appraisal(d) + _trust(d) + _place(d) + _full_text(d)
    out += ["## Cite this paper", "", p.apa, "", "```bibtex", p.bibtex, "```", ""]
    return "\n".join(out) + "\n"


def _header(d: PaperDetail) -> list[str]:
    p = d.paper
    links = []
    if p.paper.doi:
        links.append(f"[DOI](https://doi.org/{p.paper.doi})")
    if p.paper.pdf_url:
        links.append(f"[PDF]({p.paper.pdf_url})")
    if p.paper.landing_url and not p.paper.doi:
        links.append(f"[Page]({p.paper.landing_url})")
    facts = [*links, str(p.paper.year or "year unknown"), STUDY_NAMES.get(p.study_type, p.study_type),
             f"cited {p.paper.citation_count or 0} times"]
    ordered = sorted(p.score_parts.items(), key=lambda kv: PART_ORDER.index(kv[0]) if kv[0] in PART_ORDER else 99)
    parts = " + ".join(f"{name.replace('_', ' ')} {value:.2f}" for name, value in ordered)
    return [f"# {p.paper.title}", "", p.apa, "", " · ".join(facts), "",
            f"**In topic #{d.topic_id} \"{d.topic}\":** rank {p.rank} of {d.n_papers} · relevance "
            f"{p.relevance}/10 ({p.relevance_reason}) · score {p.score:.2f} = {parts}", ""]


def _summary(p: ReportPaper) -> list[str]:
    s = p.summary
    if s is None:
        return ["## Summary", "", "_Summary unavailable: the LLM call failed._", ""]
    if s.summary is None:
        return ["## Summary", "", "_No abstract and no free full text: only the title is known._", ""]
    b = s.summary
    out = [f"## Summary (from the {s.based_on})", "", f"**TL;DR:** {b.tldr}", "",
           f"- **Objective:** {b.objective}", f"- **Methods:** {b.methods}", "", "**Key findings**", ""]
    for f in b.key_findings:
        out.append(f"- {f.statement}")
        if f.quote:
            out.append(f'  > "{f.quote}"')
    out += [""] + [f"> ⚠ {flag}" for flag in s.flags] + ([""] if s.flags else [])
    limitations = "; ".join(b.limitations) or "none stated"
    return out + [f"**Limitations stated by the authors:** {limitations}", "",
                  f"**Why it's relevant:** {b.why_relevant}", ""]


def _claims(p: ReportPaper) -> list[str]:
    if p.analysis is None:
        return ["## Claims", "", "_Not analysed yet: run `python -m app.cli analyze <topic id>`._", ""]
    out = ["## Claims", "", "| # | Result | Population | Evidence from the paper |", "|---|---|---|---|"]
    for i, c in enumerate(p.analysis.analysis.claims, 1):
        magnitude = f" ({c.magnitude})" if c.magnitude else ""
        out.append(f"| {i} | {ARROWS[c.direction]} {_cell(c.factor)} → {_cell(c.outcome)}{_cell(magnitude)} | "
                   f"{_cell(c.population)} | \"{_cell(c.quote)}\" |")
    unverified = [f for f in p.analysis.flags if f.startswith("claim ")]
    return out + [""] + [f"> ⚠ {flag}" for flag in unverified] + ([""] if unverified else [])


def _appraisal(d: PaperDetail) -> list[str]:
    p = d.paper
    if p.analysis is None:
        return ["## Quality appraisal", "", "_Not analysed yet._", ""]
    a = p.analysis.analysis.appraisal
    n = a.sample_size
    out = ["## Quality appraisal", "", "| Item | Value | Evidence from the paper |", "|---|---|---|",
           f"| Sample size | {n if n is not None else '?'}{' (small)' if n is not None and n < 10 else ''} | "
           f"{_quote(a.sample_size_quote)} |",
           f"| Sex · duration | {a.sex} · {a.duration} | |"]
    for key, name in CHECKLIST.items():
        fact = getattr(a, key)
        if fact.value == "not applicable":
            continue
        evidence = _quote(fact.quote) if fact.value in ("yes", "no") else ""
        if any(flag.startswith(FACT_FLAG.format(name=key)) for flag in p.analysis.flags):
            evidence = "⚠ the AI said yes but could not quote it, so it is not counted"
        out.append(f"| {name.capitalize()} | {MARKS[fact.value]} {fact.value} | {evidence} |")
    earned, possible, _ = quality_score(a, d.field_family)             # the same code that set the trust
    share = f" ({earned / possible:.0%})" if possible else ""
    return out + ["", f"**Score:** {earned:g} of {possible:g} applicable points{share} · judged from the "
                      f"{p.analysis.based_on} _(✓ shown with a quote · ✗ stated not done · ? not reported)_", ""]


def _trust(d: PaperDetail) -> list[str]:
    p = d.paper
    base = p.base_trust or p.trust
    out = [f"## Trust: {_trust_text(p)}", "",
           f"1. **From the study design:** {base.label} ({base.level}/5): {'; '.join(base.reasons)}"]
    added = p.trust.reasons[len(base.reasons):]                       # what the quality check (5.2) added
    if not added:
        out.append("2. **From the quality check:** not analysed yet" if p.analysis is None
                   else "2. **From the quality check:** no change")
    else:
        note = added[-1]
        checklist, _, red_flag = note.partition("; very small sample")
        out.append(f"2. **From the quality check:** {checklist}")
        if red_flag:
            out.append(f"3. **Red flag:** very small sample{red_flag}")
    return out + ["", f"**Result: {p.trust.label} ({p.trust.level}/5)**", ""]


def _place(d: PaperDetail) -> list[str]:
    me = d.paper.paper_id
    out = ["## Place in this topic", ""]
    if not d.connections:
        return out + ["_No other selected paper answers the same questions as this one._", ""]
    for c in d.connections:
        icon, word = RELATIONS[c.relation]
        out += [f"### {icon} {word}: {c.question}", ""]
        mine = [m for m in c.members if m.paper == me]
        others = [m for m in c.members if m.paper != me]
        for m in mine + others:
            claim = m.claim_snapshot
            if claim is None:
                continue
            # other papers link to their section in the topic report (same folder, predictable file name)
            who = "**this paper**" if m.paper == me else \
                f"[{d.labels.get(m.paper, '?')}]({d.topic_id}-{slugify(d.topic)}.md#{d.ranks.get(m.paper, '')})"
            magnitude = f" ({claim.magnitude})" if claim.magnitude else ""
            out.append(f"- {ARROWS[claim.direction]} {who} {m.stance}: {claim.outcome}{magnitude}")
        if c.why_differ:
            out += ["", f"**Why they may differ:** {c.why_differ}"]
        out.append("")
    return out


def _full_text(d: PaperDetail) -> list[str]:
    out = ["## Full text used", ""]
    doc = d.document
    if doc is None:
        return out + ["_Summarised from the abstract only: no free full text was found._", ""]
    kind = "XML (structured, from Europe PMC)" if doc.content_type == "application/xml" else "PDF"
    pages = f" · {doc.page_count} pages" if doc.page_count else ""
    source = f" from {doc.source_url}" if doc.source_url else ""
    out.append(f"{kind}{source} · {doc.size_bytes // 1024} KB{pages}")
    if doc.status != "extracted":
        return out + ["", f"_Not usable: {doc.error or doc.status}. Summarised from the abstract instead._", ""]
    totals: dict[str, int] = {}
    for s in doc.sections:
        totals[s.kind] = totals.get(s.kind, 0) + s.characters
    ordered = sorted(totals.items(), key=lambda kv: KIND_ORDER.index(kv[0]) if kv[0] in KIND_ORDER else 99)
    return out + ["", f"Extracted into {len(doc.sections)} sections: "
                      + " · ".join(f"{kind} {chars / 1000:.1f}k" for kind, chars in ordered) + " characters", ""]


def _quote(text: str | None) -> str:
    return f'"{_cell(text)}"' if text else ""
