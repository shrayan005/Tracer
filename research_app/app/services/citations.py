"""
Citations (step 3.5): APA 7 references, in-text citations and BibTeX, built by CODE from metadata.

Why not the LLM? Citations must be exact. The LLM might invent a co-author, a year or a DOI;
code only formats what the sources actually returned, so it can never make one up.

Author names arrive in different shapes depending on the source:
  "John A. Smith" (OpenAlex, arXiv, Semantic Scholar)   "Smith JA" (Europe PMC)   "Smith, John A."
split_name() turns each into (family name, given names / initials).
"""

import re
import unicodedata

from app.domain.schemas import PaperRecord

APA_MAX_AUTHORS = 20        # APA 7: list up to 20 authors; more -> first 19, "...", last
BIBTEX_SPECIAL = re.compile(r"([&%$#_])")


def split_name(name: str) -> tuple[str, str]:
    """Return (family, given). `given` may be full names ("John A.") or initials ("J A")."""
    name = " ".join(name.split())
    if "," in name:                                    # "Smith, John A."
        family, given = name.split(",", 1)
        return family.strip(), given.strip()
    parts = name.split(" ")
    if len(parts) == 1:
        return parts[0], ""
    last = parts[-1].replace(".", "")
    if last.isupper() and len(last) <= 3:              # Europe PMC "Smith JA" -> initials at the end
        return " ".join(parts[:-1]), " ".join(last)
    return parts[-1], " ".join(parts[:-1])             # "John A. Smith"


def initials(given: str) -> str:
    """'John Albert' -> 'J. A.'   'J A' -> 'J. A.'   'Jean-Paul' -> 'J.-P.'"""
    out = []
    for word in given.replace(".", " ").split():
        out.append("-".join(f"{piece[0].upper()}." for piece in word.split("-") if piece))
    return " ".join(out)


def apa_author(name: str) -> str:
    family, given = split_name(name)
    return f"{family}, {initials(given)}" if given else family


def apa_authors(authors: list[str]) -> str:
    names = [apa_author(a) for a in authors if a.strip()]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) > APA_MAX_AUTHORS:
        return ", ".join(names[:APA_MAX_AUTHORS - 1]) + ", . . . " + names[-1]
    return ", ".join(names[:-1]) + ", & " + names[-1]


def link(paper: PaperRecord) -> str | None:
    if paper.doi:
        return f"https://doi.org/{paper.doi}"
    if paper.arxiv_id:
        return f"https://arxiv.org/abs/{paper.arxiv_id}"
    return paper.landing_url


def apa(paper: PaperRecord) -> str:
    """APA 7 reference (Markdown: venue in *italics*). Titles are kept as the source gives them."""
    year = f"({paper.year})" if paper.year else "(n.d.)"
    title = paper.title.rstrip(".")
    authors = apa_authors(paper.authors)
    head = f"{authors} {year}. {title}." if authors else f"{title}. {year}."
    parts = [head]
    if paper.venue:
        parts.append(f"*{paper.venue}*.")
    if url := link(paper):
        parts.append(url)
    return " ".join(parts)


def in_text(paper: PaperRecord) -> str:
    """(Smith, 2022)   (Smith & Lee, 2022)   (Smith et al., 2022)"""
    families = [split_name(a)[0] for a in paper.authors if a.strip()]
    year = paper.year or "n.d."
    if not families:
        short = paper.title if len(paper.title) <= 40 else paper.title[:40].rsplit(" ", 1)[0] + "..."
        return f'("{short}", {year})'
    if len(families) == 1:
        return f"({families[0]}, {year})"
    if len(families) == 2:
        return f"({families[0]} & {families[1]}, {year})"
    return f"({families[0]} et al., {year})"


def _ascii(text: str) -> str:
    """'Müller' -> 'muller' (BibTeX keys must be plain ASCII)."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", plain.lower())


def bibtex_key(paper: PaperRecord) -> str:
    """smith2022triceps: first author's family name + year + first meaningful title word."""
    family = _ascii(split_name(paper.authors[0])[0]) if paper.authors else "anon"
    words = [_ascii(w) for w in paper.title.split()]
    word = next((w for w in words if len(w) > 3), "paper")
    return f"{family or 'anon'}{paper.year or 'nd'}{word}"


def unique_bibtex_keys(papers: list[PaperRecord]) -> list[str]:
    """Same key twice (smith2022triceps) -> smith2022tricepsa, smith2022tricepsb."""
    keys = [bibtex_key(p) for p in papers]
    seen: dict[str, int] = {}
    out = []
    for key in keys:
        if keys.count(key) > 1:
            seen[key] = seen.get(key, 0) + 1
            key = key + "abcdefghijklmnopqrstuvwxyz"[seen[key] - 1]
        out.append(key)
    return out


def bib_author(name: str) -> str:
    """'John A. Smith' -> 'Smith, John A.'   'Smith JA' -> 'Smith, J. A.' (initials get dots)."""
    family, given = split_name(name)
    if not given:
        return family
    if all(len(w) == 1 for w in given.replace(".", " ").split()):
        given = initials(given)
    return f"{family}, {given}"


def _bib_escape(text: str) -> str:
    return BIBTEX_SPECIAL.sub(r"\\\1", text)


def bibtex(paper: PaperRecord, key: str | None = None) -> str:
    """One BibTeX entry. Preprints with an arXiv id become @misc with eprint fields."""
    is_preprint = paper.publication_type == "preprint" or paper.venue_type == "preprint"
    entry_type = "misc" if is_preprint else "article"
    authors = " and ".join(bib_author(a) for a in paper.authors if a.strip())
    fields = {
        "title": "{" + _bib_escape(paper.title) + "}",      # double braces keep capitals (DNA, EMG)
        "author": _bib_escape(authors) or None,
        "journal": _bib_escape(paper.venue) if paper.venue and not is_preprint else None,
        "year": str(paper.year) if paper.year else None,
        "doi": paper.doi,
        "eprint": paper.arxiv_id,
        "archivePrefix": "arXiv" if paper.arxiv_id else None,
        "url": None if paper.doi else link(paper),
    }
    lines = [f"  {name} = {{{value}}}," for name, value in fields.items() if value]
    return f"@{entry_type}{{{key or bibtex_key(paper)},\n" + "\n".join(lines) + "\n}"
