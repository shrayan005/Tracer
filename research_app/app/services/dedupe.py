"""
Merge duplicate papers found by different sources and queries.

The same paper often comes back 3-6 times (OpenAlex + Europe PMC + several queries).
Two records are the same paper if they share ANY identifier (DOI, PubMed id, arXiv id,
OpenAlex id, Semantic Scholar id) or, as a last resort, the same normalised title.
Pure functions: no database, no network, easy to unit-test.
"""

import re

from app.domain.schemas import Candidate, FoundBy, PaperRecord

ID_KEYS = ("doi", "pmid", "arxiv_id", "openalex_id", "s2_id")
MIN_TITLE_KEY_LENGTH = 25   # short titles like "Introduction" must not merge different papers


def normalize_title(title: str) -> str:
    """'Triceps EMG: A Study!' -> 'triceps emg a study'"""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", title.lower())).strip()


def record_keys(record: PaperRecord) -> list[str]:
    """Every way to recognise this paper, e.g. ['doi:10.1/a', 'pmid:99', 'title:triceps ...']."""
    keys = [f"{name}:{getattr(record, name)}" for name in ID_KEYS if getattr(record, name)]
    title = normalize_title(record.title)
    if len(title) >= MIN_TITLE_KEY_LENGTH:
        keys.append(f"title:{title}")
    return keys


def merge_records(records: list[PaperRecord]) -> list[Candidate]:
    """Group records that are the same paper; combine their data and remember who found them."""
    candidates: list[Candidate] = []
    index: dict[str, int] = {}          # key ("doi:10.1/x") -> position in candidates

    for record in records:
        keys = record_keys(record)
        match = next((index[k] for k in keys if k in index), None)   # seen any of these keys before?
        found = FoundBy(source=record.source, query=record.found_by_query or "")

        if match is None:                                  # new paper
            candidates.append(Candidate(paper=record.model_copy(), found_by=[found]))
            match = len(candidates) - 1
        else:                                              # duplicate → merge into the existing one
            candidate = candidates[match]
            candidate.paper = merge_two(candidate.paper, record)
            if found not in candidate.found_by:
                candidate.found_by.append(found)

        # register every key of this record AND of the merged paper (it may have gained new ids)
        for key in keys + record_keys(candidates[match].paper):
            index.setdefault(key, match)
    return candidates


def merge_two(base: PaperRecord, other: PaperRecord) -> PaperRecord:
    """Combine two records of the same paper: fill gaps, keep the longest abstract, max counts."""
    data = base.model_dump()
    for name, value in other.model_dump().items():
        if name in ("source", "found_by_query"):           # keep the first source's values
            continue
        if data.get(name) in (None, "", []) and value not in (None, "", []):
            data[name] = value                             # fill a missing value
    if other.abstract and len(other.abstract) > len(data.get("abstract") or ""):
        data["abstract"] = other.abstract                  # longer abstract = more information
    for count in ("citation_count", "influential_citation_count"):
        values = [v for v in (getattr(base, count), getattr(other, count)) if v is not None]
        data[count] = max(values) if values else None      # sources update at different times
    data["is_retracted"] = base.is_retracted or other.is_retracted          # one warning is enough
    data["pub_types"] = base.pub_types + [t for t in other.pub_types if t not in base.pub_types]   # all tags
    data["is_open_access"] = True if (base.is_open_access or other.is_open_access) else data["is_open_access"]
    return PaperRecord(**data)
