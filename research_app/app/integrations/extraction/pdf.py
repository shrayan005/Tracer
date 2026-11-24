"""
PDF extraction with PyMuPDF + pymupdf4llm: fixes the reading order of two-column pages, marks
headings with '#', turns tables into Markdown tables. Then the Markdown is cut into sections.

LICENSE: PyMuPDF is AGPL-3.0 (or a paid Artifex licence). It is used ONLY in this file, so before
the app is opened to other users it can be replaced (e.g. by GROBID, Apache-2.0) without touching
anything else. Everything else only sees extract_pdf() -> Extraction.

The PDF comes from outside: refuse password-protected, oversized and image-only (scanned) files.
OCR is switched off: it is slow and would need extra software; scanned papers fall back to the abstract.
"""

import pymupdf
import pymupdf4llm

from app.domain.errors import ExtractionError
from app.integrations.extraction.sections import Extraction, split_markdown

MIN_CHARS_PER_PAGE = 200          # less text than this on average = pictures of pages, not text


def extract_pdf(data: bytes, max_pages: int = 100) -> Extraction:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise ExtractionError(f"cannot open PDF: {type(e).__name__}") from e
    with doc:
        if doc.needs_pass:
            raise ExtractionError("PDF is password-protected")
        if doc.page_count == 0 or doc.page_count > max_pages:
            raise ExtractionError(f"unsupported page count: {doc.page_count} (max {max_pages})")
        chars = sum(len(page.get_text()) for page in doc)
        if chars < MIN_CHARS_PER_PAGE * doc.page_count:
            raise ExtractionError("no text layer (scanned PDF): would need OCR")
        markdown = pymupdf4llm.to_markdown(doc, use_ocr=False, header=False, footer=False,
                                           show_progress=False)
        return Extraction(split_markdown(markdown), page_count=doc.page_count)
