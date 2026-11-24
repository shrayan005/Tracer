import pytest

from app.domain.schemas import PaperRecord
from app.services.citations import (apa, apa_authors, bibtex, bibtex_key, in_text, initials, split_name,
                                    unique_bibtex_keys)


def paper(**kwargs) -> PaperRecord:
    defaults = {"source": "x", "title": "Triceps activation during bench press", "year": 2022}
    return PaperRecord(**{**defaults, **kwargs})


@pytest.mark.parametrize("name, expected", [
    ("John A. Smith", ("Smith", "John A.")),          # OpenAlex / arXiv / Semantic Scholar
    ("Smith JA", ("Smith", "J A")),                   # Europe PMC
    ("Smith, John A.", ("Smith", "John A.")),
    ("van der Berg P", ("van der Berg", "P")),
    ("Plato", ("Plato", "")),
    ("  John   Smith ", ("Smith", "John")),
])
def test_split_name(name, expected):
    assert split_name(name) == expected


def test_initials():
    assert initials("John Albert") == "J. A."
    assert initials("J A") == "J. A."
    assert initials("Jean-Paul") == "J.-P."


def test_apa_author_lists():
    assert apa_authors(["John Smith"]) == "Smith, J."
    assert apa_authors(["John Smith", "Kim Lee"]) == "Smith, J., & Lee, K."
    many = apa_authors([f"Ann Author{i}" for i in range(25)])
    assert many.startswith("Author0, A., ") and ", . . . Author24, A." in many
    assert "Author19" not in many                       # first 19, then ". . .", then the last


def test_apa_full_reference_prefers_doi():
    ref = apa(paper(authors=["John Smith", "Kim Lee"], venue="J Biomech", doi="10.1/abc",
                    landing_url="https://example.org/page"))
    assert ref == ("Smith, J., & Lee, K. (2022). Triceps activation during bench press. "
                   "*J Biomech*. https://doi.org/10.1/abc")


def test_apa_without_authors_or_year():
    assert apa(paper(year=None)) == "Triceps activation during bench press. (n.d.)."


def test_in_text():
    assert in_text(paper(authors=["John Smith"])) == "(Smith, 2022)"
    assert in_text(paper(authors=["John Smith", "Kim Lee"])) == "(Smith & Lee, 2022)"
    assert in_text(paper(authors=["John Smith", "Kim Lee", "Ann Park"])) == "(Smith et al., 2022)"
    assert in_text(paper()) == '("Triceps activation during bench press", 2022)'


def test_bibtex_keys_are_ascii_and_unique():
    p = paper(authors=["Jürgen Müller"])
    assert bibtex_key(p) == "muller2022triceps"
    other = paper(authors=["Kim Lee"], title="Elbow torque")
    assert unique_bibtex_keys([p, other, p]) == ["muller2022tricepsa", "lee2022elbow", "muller2022tricepsb"]


def test_bibtex_article():
    entry = bibtex(paper(authors=["John A. Smith", "Lee K"], venue="J Biomech", doi="10.1/abc"))
    assert entry.startswith("@article{smith2022triceps,")
    assert "author = {Smith, John A. and Lee, K.}," in entry
    assert "journal = {J Biomech}," in entry and "doi = {10.1/abc}," in entry
    assert "url" not in entry                           # the DOI already is the link


def test_bibtex_preprint_and_escaping():
    entry = bibtex(paper(title="EMG & force: 50% faster", publication_type="preprint", arxiv_id="2501.01234",
                         venue="arXiv"))
    assert entry.startswith("@misc{")
    assert "title = {{EMG \\& force: 50\\% faster}}," in entry
    assert "eprint = {2501.01234}," in entry and "archivePrefix = {arXiv}," in entry
    assert "journal" not in entry
