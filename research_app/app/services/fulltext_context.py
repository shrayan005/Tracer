"""
Which parts of a paper go into the summary call (step 4.5).

A paper is 30-80k characters (~10-20k tokens). Most of the value is in methods, results, discussion
and limitations, so each kind gets a character budget (policy as data, ~5k tokens in total).
Long parts are cut at a sentence end and marked [...]. References, author lists, funding and
figure captions are left out.

Authors usually admit limitations near the END of the Discussion, exactly the part a trim cuts off,
so paragraphs that talk about limitations are moved into "limitations" before trimming.
"""

import re
from dataclasses import dataclass, field

# kind -> max characters (about 4 characters per token); also the order in the prompt
BUDGET = {"abstract": 1_500, "methods": 5_000, "results": 5_000, "table": 2_000,
          "discussion": 3_000, "limitations": 1_500, "conclusion": 1_500}
SENTENCE_END = re.compile(r"(?<=[.!?])\s")
STATEMENT_WORDS = re.compile(r"conflicts? of interest|competing interests?|\bfunding\b|\bfunded by\b|"
                             r"grant support|data availability|original contributions presented|"
                             r"\bdata(?:sets?)?\b[^.]{0,80}\b(?:publicly|freely|openly|made available|"
                             r"available (?:at|from|on|upon))", re.IGNORECASE)
STATISTICS_TITLE = re.compile(r"statistic|data analys", re.IGNORECASE)
STATEMENT_MAX_CHARS = 400           # one statement sentence; longer = extraction junk around it
LIMITATION_WORDS = re.compile(r"\blimitation|\blimited by\b|\bshould be interpreted with caution\b|"
                              r"\bfuture (?:research|studies|work)\b|\bnot generali[sz]", re.IGNORECASE)


@dataclass
class PaperText:
    context: str        # the trimmed text sent to the summary LLM call
    source: str         # ALL extracted text: quotes and numbers are checked against this
    sections: list[tuple[str, str | None, str]] = field(default_factory=list)   # to build other contexts


def trim(text: str, limit: int) -> str:
    """Cut to `limit` characters at a sentence end (or a space), marking the cut with [...]."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    ends = [m.end() for m in SENTENCE_END.finditer(cut)]
    if ends and ends[-1] > limit * 0.6:
        cut = cut[:ends[-1]]
    elif " " in cut:
        cut = cut[:cut.rfind(" ")]
    return cut.rstrip() + " [...]"


def build_context(sections: list[tuple[str, str | None, str]], budget: dict[str, int] = BUDGET) -> str:
    """(kind, title, text) sections in reading order -> one text with '## Kind' headings, within budget.
    Each task can pass its own budget (the analysis needs more methods, plus the backmatter)."""
    by_kind: dict[str, list[str]] = {}
    if "statements" in budget:
        by_kind["statements"] = statements(sections)
    for kind, title, text in sections:
        if kind == "methods" and "statistics" in budget and STATISTICS_TITLE.search(title or ""):
            kind = "statistics"          # the LAST methods subsection: a methods trim would cut it (6.7)
        if kind in ("discussion", "conclusion"):
            paragraphs = [p for p in text.split("\n\n") if p.strip()]
            admitted = [p for p in paragraphs if LIMITATION_WORDS.search(p)]
            by_kind.setdefault("limitations", []).extend(admitted)
            text = "\n\n".join(p for p in paragraphs if p not in admitted)
        if kind in budget and text.strip():
            heading = f"### {title}\n" if kind == "table" and title else ""
            by_kind.setdefault(kind, []).append(heading + text)
    return "\n\n".join(f"## {kind.capitalize()}\n{trim(chr(10).join(by_kind[kind]), limit)}"
                       for kind, limit in budget.items() if by_kind.get(kind))


def statements(sections: list[tuple[str, str | None, str]]) -> list[str]:
    """Conflict-of-interest, funding and data-availability sentences from ANY section (not the references),
    so a trim can't cut them: they often sit at the very end of a paper or under 'Conclusions'."""
    found: list[str] = []
    for kind, _title, text in sections:
        if kind == "references":
            continue
        for sentence in SENTENCE_END.split(text):
            sentence = " ".join(sentence.split())
            match = STATEMENT_WORDS.search(sentence)
            if not match:
                continue
            if len(sentence) > STATEMENT_MAX_CHARS:           # junk around it (PDF): keep the part at the match
                start = max(0, match.start() - 100)
                sentence = sentence[start:start + STATEMENT_MAX_CHARS]
            if sentence not in found:
                found.append(sentence)
    return found


def paper_text(sections: list[tuple[str, str | None, str]]) -> PaperText:
    return PaperText(context=build_context(sections), source="\n\n".join(text for _, _, text in sections),
                     sections=list(sections))
