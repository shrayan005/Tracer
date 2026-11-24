"""
Passages for questions (step 7.6a): a paper's sections -> passages of ~1,000 characters.

A passage is what the search returns and what an answer quotes from, so it must be small enough to be
specific and big enough to hold a whole finding. Cut at sentence ends, never mid-sentence; the last
sentence of a passage is repeated at the start of the next (overlap), so a finding that spans the cut
is still whole in one of them. Pure code: unit-tested.
"""

import re
from dataclasses import dataclass

CHUNKER_VERSION = "1"            # raise it when the rules change: papers are re-indexed
TARGET, MAX = 1000, 1500         # characters per passage (a paragraph or two)
MIN_TEXT = 80                    # shorter section text is noise (a heading, a page number)
SKIP_KINDS = {"references", "front", "backmatter", "caption"}   # author lists, citations, licences, figure labels
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'0-9])")


@dataclass
class Passage:
    kind: str                    # section kind: abstract, methods, results, table, ...
    title: str | None            # section title, e.g. "2.3 Statistical analysis"
    position: int                # order in the paper (0, 1, 2 ...)
    text: str


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(" ".join(text.split())) if s.strip()]


def split_text(text: str) -> list[str]:
    """One section's text -> passages of about TARGET characters, cut at sentence ends, 1-sentence overlap."""
    parts: list[str] = []
    current: list[str] = []
    fresh = False                                        # does `current` hold a sentence not yet in a passage?
    for sentence in sentences(text):
        while len(sentence) > MAX:                       # one huge "sentence" (a table row dump): hard cut
            parts.append(sentence[:MAX])
            sentence = sentence[MAX:]
        if fresh and len(" ".join(current)) + len(sentence) > TARGET:
            parts.append(" ".join(current))
            current = [current[-1]] if len(current[-1]) < TARGET // 2 else []   # overlap: the last sentence
        current.append(sentence)
        fresh = True
    if fresh:
        parts.append(" ".join(current))
    return parts


def chunk_paper(sections: list[tuple[str, str | None, str]], abstract: str | None) -> list[Passage]:
    """All passages of one paper. Full text when there is any; the abstract always (a paper with no full text
    is still searchable by its abstract)."""
    passages: list[Passage] = []
    has_abstract = any(kind == "abstract" for kind, _, _ in sections)
    if abstract and not has_abstract and len(abstract.strip()) >= MIN_TEXT:
        sections = [("abstract", "Abstract", abstract), *sections]
    for kind, title, text in sections:
        if kind in SKIP_KINDS or len((text or "").strip()) < MIN_TEXT:
            continue
        for part in split_text(text):
            passages.append(Passage(kind=kind, title=title, position=len(passages), text=part))
    return passages
