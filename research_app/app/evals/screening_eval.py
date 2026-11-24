"""
Screening eval (step 6.5, part A): is "relevant" judged right?

    prefill   evals/gold/screening.yaml: every screened candidate of each case with the AI's score and a
              pre-filled bucket (relevant 6-10 / partly 3-5 / not relevant 0-2) for YOU to correct
    suite     re-screens the frozen candidates with the CURRENT prompt and model and compares with your buckets

The worst screening error is a missed relevant paper (it never reaches the report), so
`relevant_recall` is the headline metric.
"""

from app.domain.schemas import PaperRecord
from app.evals.cases import EvalCase
from app.evals.labels import BUCKETS, SCREENING, load_screening_gold, write_yaml
from app.evals.runner import Services, SuiteResult, suite
from app.services.research_service import with_official_type
from app.services.selection import MIN_RELEVANCE

PARTLY_FROM = 3
ABSTRACT_CHARS = 400


def bucket(relevance: int) -> str:
    if relevance >= MIN_RELEVANCE:
        return "relevant"
    return "partly" if relevance >= PARTLY_FROM else "not relevant"


def link(paper: PaperRecord) -> str | None:
    if paper.doi:
        return f"https://doi.org/{paper.doi}"
    return paper.landing_url


def prefill_screening(cases: list[EvalCase], force: bool = False) -> tuple[str, int]:
    """Write the label file. Returns (path, number of papers). Never overwrites labels by accident."""
    if SCREENING.exists() and not force:
        raise FileExistsError(f"{SCREENING} already exists and may hold your labels (use --force to replace it)")
    blocks, total = [], 0
    for case in cases:
        if not case.candidates:
            continue                                   # frozen before 6.5: re-freeze with --force
        papers = []
        for c in case.candidates:
            abstract = (c.paper.abstract or "(no abstract)").strip()
            papers.append({"key": c.key, "title": c.paper.title, "year": c.paper.year, "link": link(c.paper),
                           "abstract": abstract[:ABSTRACT_CHARS] + ("..." if len(abstract) > ABSTRACT_CHARS else ""),
                           "ai_relevance": c.relevance, "ai_reason": c.reason, "ai_study_type": c.study_type,
                           "gold": bucket(c.relevance), "gold_study_type": None})
        total += len(papers)
        blocks.append({"case": case.name, "brief": case.brief, "reviewed": False, "papers": papers})
    if not blocks:
        raise ValueError("no case has screened candidates: re-freeze with `python -m app.evals freeze <id> <name> --force`")
    how_to = ("For each paper set `gold:` to one of: " + ", ".join(BUCKETS) + " (for the case's brief). "
              "Optionally set `gold_study_type:` (e.g. randomized_trial) to check study types too. "
              "Then set `reviewed: true` for the case.")
    write_yaml(SCREENING, {"how_to": how_to, "cases": blocks})
    return str(SCREENING), total


def share(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


@suite("screening")
async def screening(cases: list[EvalCase], services: Services) -> SuiteResult:
    """Re-screen each reviewed case's candidates and compare with your buckets."""
    if services.screener is None:
        raise ValueError("the screening suite needs a screener")
    gold = load_screening_gold()
    rows, details = [], []                      # (gold bucket, ai bucket, gold type, checked type, LLM's type)
    for case in cases:
        labels = gold.get(case.name)
        if not labels:
            continue
        candidates = [c for c in case.candidates if c.key in labels]
        results = await services.screener.screen(case.brief, [c.paper for c in candidates])
        for i, c in enumerate(candidates):
            label = labels[c.key]
            s = results.get(i)
            if s is None:
                details.append(f"{case.name} · {c.paper.title[:70]}: not screened (batch failed)")
                continue
            ai = bucket(s.relevance)
            llm_type = s.study_type
            s = with_official_type(s, c.paper)                 # the pipeline's check against index tags
            rows.append((label["gold"], ai, label["study_type"], s.study_type, llm_type))
            if label["gold"] == "relevant" and ai != "relevant":
                details.append(f"MISSED · {case.name} · {c.paper.title[:80]} → AI {s.relevance}/10: {s.reason}")
            elif ai == "relevant" and label["gold"] != "relevant":
                details.append(f"FALSE RELEVANT · {case.name} · {c.paper.title[:80]} → AI {s.relevance}/10, "
                               f"you: {label['gold']}")
            if label["study_type"] and label["study_type"] != s.study_type:
                details.append(f"STUDY TYPE · {case.name} · {c.paper.title[:70]}: AI {s.study_type}, "
                               f"you {label['study_type']}")
    if not rows:
        raise ValueError("no reviewed screening labels for these cases")
    gold_relevant = [r for r in rows if r[0] == "relevant"]
    ai_relevant = [r for r in rows if r[1] == "relevant"]
    typed = [r for r in rows if r[2]]
    metrics = {
        "papers": float(len(rows)),
        "relevant_recall": share(sum(r[1] == "relevant" for r in gold_relevant), len(gold_relevant)),
        "relevant_precision": share(sum(r[0] == "relevant" for r in ai_relevant), len(ai_relevant)),
        "bucket_accuracy": share(sum(r[0] == r[1] for r in rows), len(rows)),
        "gold_relevant": float(len(gold_relevant)),
    }
    if typed:
        metrics["study_type_accuracy"] = share(sum(r[2] == r[3] for r in typed), len(typed))
        metrics["study_type_accuracy_llm_only"] = share(sum(r[2] == r[4] for r in typed), len(typed))
    return metrics, details
