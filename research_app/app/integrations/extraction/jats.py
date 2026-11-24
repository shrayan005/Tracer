"""
JATS XML extraction: Europe PMC / PubMed Central full texts. The structure is already marked up
(<sec><title>Methods</title><p>...</p></sec>, <table-wrap>, <fig>), so no guessing is needed.

Parsed with defusedxml: the XML comes from outside, and the standard parser can be tricked by
"entity" definitions (one 1 KB file expanding to gigabytes, or reading files from the server).
"""

import re

import defusedxml.ElementTree as ET

from app.domain.errors import ExtractionError
from app.integrations.extraction.sections import Extraction, Section, classify


def _text(element) -> str:
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def _table_markdown(table_wrap) -> str:
    rows = []
    for tr in table_wrap.iter("tr"):
        cells = [_text(cell).replace("|", "\\|") for cell in tr if cell.tag in ("th", "td")]
        if cells:
            rows.append("| " + " | ".join(cells) + " |")
    if len(rows) > 1:
        rows.insert(1, "|" + "---|" * (rows[0].count(" | ") + 1))     # header separator
    return "\n".join(rows)


def _section_text(sec, depth: int = 0) -> str:
    """Paragraphs of a <sec>, with sub-section titles kept as small headings."""
    parts = []
    for child in sec:
        if child.tag == "p":
            parts.append(_text(child))
        elif child.tag == "sec":
            sub_title = (child.findtext("title") or "").strip()
            if sub_title:
                parts.append(f"{'#' * min(depth + 3, 6)} {sub_title}")
            parts.append(_section_text(child, depth + 1))
    return "\n\n".join(p for p in parts if p)


def extract_jats(data: bytes) -> Extraction:
    try:
        root = ET.fromstring(data)
    except Exception as e:                      # malformed XML or a refused entity trick
        raise ExtractionError(f"invalid XML: {type(e).__name__}") from e

    sections: list[Section] = []
    abstract = root.find(".//front//abstract")
    if abstract is not None and _text(abstract):
        sections.append(Section("abstract", "Abstract", _section_text(abstract) or _text(abstract)))

    body = root.find(".//body")
    if body is None:
        raise ExtractionError("XML has no article body (abstract only)")
    current = "other"
    for sec in body.findall("sec"):
        title = (sec.findtext("title") or "").strip() or None
        # the title decides; JATS sec-type ("materials|methods") is the fallback (it's often imprecise)
        current = classify(title) or classify(sec.get("sec-type", "").replace("|", " ")) or current
        text = _section_text(sec)
        if text:
            sections.append(Section(current, title, text))

    for wrap in root.iter("table-wrap"):
        caption = wrap.find("caption")
        label = " ".join(filter(None, [wrap.findtext("label"), _text(caption) if caption is not None else None]))
        table = _table_markdown(wrap)
        if table:
            sections.append(Section("table", label or None, table))
    for fig in root.iter("fig"):
        caption = fig.find("caption")
        if caption is not None and _text(caption):
            sections.append(Section("caption", fig.findtext("label"), _text(caption)))

    if not any(s.kind not in ("abstract", "table", "caption") for s in sections):
        raise ExtractionError("XML body has no text sections")
    return Extraction(sections)
