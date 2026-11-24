"""Text extraction: heading rules, JATS XML and PDF (PDFs are generated inside the tests)."""

import pymupdf
import pytest

from app.domain.errors import ExtractionError
from app.integrations.extraction.jats import extract_jats
from app.integrations.extraction.pdf import extract_pdf
from app.integrations.extraction.sections import classify, clean_heading, clean_text, split_markdown


# ---------- headings ----------

@pytest.mark.parametrize("raw, clean", [
    ("## **2.1. Participants** ", "Participants"),
    ("2.5.Determination of Anthropometry", "Determination of Anthropometry"),
    ("II. RESULTS", "RESULTS"),
    ("iv) Discussion", "Discussion"),
    ("INTRODUCTION", "INTRODUCTION"),                      # the "I" is NOT a Roman numeral here
    ("<mark>METHODS</mark>", "METHODS"),
])
def test_clean_heading(raw, clean):
    assert clean_heading(raw) == clean


@pytest.mark.parametrize("title, kind", [
    ("Materials and Methods", "methods"), ("Participants", "methods"), ("Statistical analyses", "methods"),
    ("RESULTS", "results"), ("Results and Discussion", "results"), ("Discussion", "discussion"),
    ("Discussion and Conclusions", "discussion"), ("Summary", "abstract"), ("Summary and conclusions", "conclusion"),
    ("Conclusions", "conclusion"), ("Strengths and limitations", "limitations"), ("Background", "introduction"),
    ("Conflicts of Interest", "backmatter"), ("References", "references"), ("EMG analysis", None),
])
def test_classify(title, kind):
    assert classify(title) == kind


def test_split_markdown_keeps_subsections_inside_their_parent():
    md = ("# A Study of Triceps\nSmith J, Lee K\n\n## INTRODUCTION\nWhy.\n\n## 2. Methods\nWe measured.\n"
          "## EMG analysis\nSignals were filtered.\n\n## RESULTS\nForce rose <mark>23%</mark>.\n## References\n1. X")
    sections = split_markdown(md)
    assert [(s.kind, s.title) for s in sections] == [
        ("front", "A Study of Triceps"), ("introduction", "INTRODUCTION"), ("methods", "Methods"),
        ("results", "RESULTS"), ("references", "References")]
    assert "### EMG analysis\nSignals were filtered." in sections[2].text
    assert sections[3].text == "Force rose 23%."                     # HTML tags removed


def test_clean_text():
    assert clean_text("a<sup>1</sup>  \n\n\n\nb") == "a1\n\nb"


# ---------- JATS XML ----------

JATS = b"""<?xml version="1.0"?>
<!DOCTYPE article PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Archiving and Interchange DTD v1.2//EN" "JATS.dtd">
<article><front><article-meta><abstract><p>Short abstract.</p></abstract></article-meta></front>
<body>
  <sec sec-type="intro"><title>Introduction</title><p>Why we did it.</p></sec>
  <sec sec-type="materials|methods"><title>Materials and methods</title>
    <sec><title>Subjects</title><p>Seven men and 10 women.</p></sec></sec>
  <sec><title>Results</title><p>Force was higher at 60 degrees.</p>
    <table-wrap><label>Table 1</label><caption><p>Force by angle.</p></caption>
      <table><tr><th>Angle</th><th>Force</th></tr><tr><td>60</td><td>12.5</td></tr></table></table-wrap>
    <fig><label>Fig 1</label><caption><p>EMG traces.</p></caption></fig></sec>
</body><back><ref-list><ref>Ignored reference</ref></ref-list></back></article>"""


def test_jats_sections_tables_and_captions():
    sections = extract_jats(JATS).sections
    kinds = [(s.kind, s.title) for s in sections]
    assert kinds == [("abstract", "Abstract"), ("introduction", "Introduction"), ("methods", "Materials and methods"),
                     ("results", "Results"), ("table", "Table 1 Force by angle."), ("caption", "Fig 1")]
    assert "### Subjects\n\nSeven men and 10 women." in sections[2].text
    assert sections[4].text == "| Angle | Force |\n|---|---|\n| 60 | 12.5 |"
    assert "Ignored reference" not in " ".join(s.text for s in sections)


def test_jats_refuses_entity_expansion_attack():
    bomb = b'<?xml version="1.0"?><!DOCTYPE a [<!ENTITY x "xxxxxxxxxx"><!ENTITY y "&x;&x;&x;&x;&x;">]><a>&y;</a>'
    with pytest.raises(ExtractionError, match="invalid XML"):
        extract_jats(bomb)


def test_jats_without_body_is_an_error():
    with pytest.raises(ExtractionError, match="no article body"):
        extract_jats(b"<article><front><abstract><p>Only this.</p></abstract></front></article>")


# ---------- PDF ----------

def make_pdf(pages: list[list[tuple[str, int]]]) -> bytes:
    """A real PDF built in memory: each page is a list of (text, font size); big text = heading."""
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page()
        y = 72
        for text, size in lines:
            page.insert_textbox(pymupdf.Rect(72, y, 540, y + 400), text, fontsize=size,
                                fontname="hebo" if size >= 14 else "helv")
            y += size * (2 + text.count("\n")) + (len(text) // 80) * size * 1.3 + 10
    return doc.tobytes()


BODY = ("The triceps brachii was measured with surface electromyography in twenty participants "
        "while they performed elbow extension at three angles. ") * 4


def test_pdf_sections_from_a_text_pdf():
    data = make_pdf([[("Triceps Function", 20), ("Introduction", 16), (BODY, 10)],
                     [("Methods", 16), (BODY, 10), ("Results", 16), (BODY, 10)]])
    result = extract_pdf(data)
    assert result.page_count == 2
    kinds = [s.kind for s in result.sections]
    assert "introduction" in kinds and "methods" in kinds and "results" in kinds


def test_pdf_with_no_text_is_treated_as_scanned():
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(50, 50, 500, 700), fill=(0.9, 0.9, 0.9))   # a "picture" of a page
    with pytest.raises(ExtractionError, match="scanned PDF"):
        extract_pdf(doc.tobytes())


def test_pdf_limits():
    with pytest.raises(ExtractionError, match="cannot open PDF"):
        extract_pdf(b"%PDF-1.7 this is not really a pdf")
    with pytest.raises(ExtractionError, match="unsupported page count: 2"):
        extract_pdf(make_pdf([[("Intro", 16), (BODY, 10)], [(BODY, 10)]]), max_pages=1)
