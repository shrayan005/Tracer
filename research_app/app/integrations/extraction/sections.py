"""
Shared pieces of text extraction (integration layer): the Section shape, text clean-up,
and turning headings into section kinds.

Every extractor (JATS XML, PDF; later GROBID) returns the same Extraction, so the service
doesn't care where the text came from.
"""

import re
from dataclasses import dataclass, field


@dataclass
class Section:
    kind: str                  # abstract / introduction / methods / results / discussion / conclusion /
                               # limitations / references / backmatter / front / table / caption / other
    title: str | None          # the heading as written in the paper
    text: str


@dataclass
class Extraction:
    sections: list[Section] = field(default_factory=list)
    page_count: int | None = None


# Heading words -> section kind. Policy as data: first match wins, so the order matters:
# "Results and Discussion" -> results, "Discussion and Conclusions" -> discussion,
# "Conclusions and limitations" -> limitations.
EXACT_TITLES = {"summary": "abstract"}    # only a heading that is exactly "Summary" is an abstract
KIND_WORDS: list[tuple[str, tuple[str, ...]]] = [
    ("abstract", ("abstract",)),
    ("references", ("references", "bibliography", "literature cited")),
    ("limitations", ("limitation",)),
    ("results", ("result", "findings")),
    ("discussion", ("discussion",)),
    ("conclusion", ("conclusion", "concluding")),
    ("methods", ("method", "materials", "participants", "subjects", "protocol", "procedure",
                 "study design", "experimental", "data collection", "statistical analys", "measurement")),
    ("introduction", ("introduction", "background")),
    ("backmatter", ("acknowledg", "funding", "disclosure", "conflict", "author contribution", "ethics",
                    "data availability", "declaration", "supplementary", "supporting information",
                    "abbreviation")),
]
# "2.1. Methods", "2.5.Methods", "II. Methods", "iv) Methods" -> "Methods". A Roman numeral needs a
# "." / ")" / space after it, otherwise the "I" of "INTRODUCTION" would be removed as a numeral.
NUMBERING = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s*|[ivxlc]+[.)]\s*|[ivxlc]+\s+)(?=[A-Za-z])", re.IGNORECASE)
HTML_TAG = re.compile(r"</?(mark|sup|sub|span|b|i|em|strong|u)(\s[^>]*)?>", re.IGNORECASE)
MARKDOWN_MARKS = re.compile(r"[*_`#]")
HEADING = re.compile(r"^#{1,6}\s+(.*)$")


def clean_text(text: str) -> str:
    """Remove leftover HTML formatting tags and squeeze blank lines (Markdown tables are kept)."""
    text = HTML_TAG.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_heading(text: str) -> str:
    """'## **2.1. Participants** ' -> 'Participants'"""
    text = MARKDOWN_MARKS.sub("", HTML_TAG.sub("", text)).strip()
    return NUMBERING.sub("", text).strip(" :.")


def classify(title: str | None) -> str | None:
    """Heading -> section kind, or None if it names no known section (e.g. 'EMG analysis')."""
    name = (title or "").lower().strip()
    if name in EXACT_TITLES:
        return EXACT_TITLES[name]
    for kind, words in KIND_WORDS:
        if any(word in name for word in words):
            return kind
    return None


def split_markdown(markdown: str) -> list[Section]:
    """
    Cut Markdown at its headings. A heading that names no known section ('EMG analysis') is a
    sub-heading: it stays INSIDE the section before it (methods), so sub-sections keep their parent.
    Text before the first known heading (title, authors, journal) becomes kind "front".
    """
    sections: list[Section] = []
    kind, title, lines = "front", None, []

    def flush() -> None:
        text = clean_text("\n".join(lines))
        if text:
            sections.append(Section(kind, title, text))

    for line in markdown.splitlines():
        match = HEADING.match(line)
        if not match:
            lines.append(line)
            continue
        heading = clean_heading(match.group(1))
        new_kind = classify(heading)
        if new_kind is None and kind != "front":
            lines.append(f"### {heading}")
            continue
        flush()
        kind, title, lines = new_kind or "front", heading or None, []
    flush()
    return sections
