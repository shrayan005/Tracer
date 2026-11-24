"""
Search recall with "borrowed gold" (step 6.5, part B): does our search find the studies experts found?

    prefill   evals/gold/recall_<case>.yaml: the reference lists of the case's systematic reviews and
              meta-analyses (from OpenAlex), merged; a reference is pre-marked `include: true` when 2+ of
              the reviews cite it or its title shares most of the topic's words. YOU confirm the list.
    suite     runs the LIVE search again (plan + search + prefilter + adaptive screening, nothing saved)
              and follows every gold paper through the stages, so you can see WHERE papers get lost.
              It also counts what each source contributed (which sources are worth their time?).

Limits: a review's references are not exactly its included studies (your check fixes that), and
they only cover what was published before the review.
"""

import re
from collections import Counter

from app.config import get_settings
from app.domain.schemas import PaperRecord
from app.evals.cases import EvalCase
from app.evals.labels import load_recall_gold, recall_path, write_yaml
from app.evals.runner import Services, SuiteResult, suite
from app.integrations.sources.openalex import OpenAlexClient
from app.services.prefilter import keywords, prefilter
from app.services.research_service import screen_adaptive
from app.services.selection import MIN_RELEVANCE

REVIEW_TYPES = {"meta_analysis", "systematic_review"}
MIN_REVIEWS = 2                 # cited by this many reviews -> pre-marked as a key study
TITLE_SHARE = 0.6               # or: the title contains this share of the topic's words


def norm_title(title: str | None) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (title or "").lower()))


def same_paper(gold: dict, paper: PaperRecord) -> bool:
    """DOI, then OpenAlex id, then the normalised title."""
    if gold.get("doi") and paper.doi and gold["doi"].lower() == paper.doi.lower():
        return True
    if gold.get("openalex_id") and gold["openalex_id"] == paper.openalex_id:
        return True
    return bool(gold.get("title")) and norm_title(gold["title"]) == norm_title(paper.title)


def premark(title: str, topic_words: set[str], cited_by: int) -> bool:
    if cited_by >= MIN_REVIEWS:
        return True
    words = keywords([title])
    needed = max(2, round(len(topic_words) * TITLE_SHARE))
    return len(words & topic_words) >= min(needed, len(topic_words))


async def prefill_recall(case: EvalCase, openalex: OpenAlexClient, force: bool = False) -> tuple[str, int, int]:
    """Write recall_<case>.yaml. Returns (path, references, pre-marked)."""
    path = recall_path(case.name)
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists and may hold your labels (use --force to replace it)")
    reviews = [p for p in case.papers if p.study_type in REVIEW_TYPES and p.paper.openalex_id]
    if not reviews:
        raise ValueError(f"case '{case.name}' has no systematic review or meta-analysis with an OpenAlex id")
    merged: dict[str, dict] = {}
    for review in reviews:
        for ref in await openalex.references(review.paper.openalex_id):
            entry = merged.setdefault(ref.openalex_id, {
                "openalex_id": ref.openalex_id, "doi": ref.doi, "title": ref.title, "year": ref.year,
                "type": ref.publication_type, "cited_by_reviews": []})
            entry["cited_by_reviews"].append(review.label)
    topic_words = keywords([case.topic])
    papers = []
    for e in sorted(merged.values(), key=lambda e: (-len(e["cited_by_reviews"]), e["title"].lower())):
        e["include"] = premark(e["title"], topic_words, len(e["cited_by_reviews"]))
        papers.append(e)
    how_to = ("Gold = the studies a good search for this topic should find. `include: true` was pre-marked "
              f"(cited by {MIN_REVIEWS}+ reviews, or the title matches the topic). Set include: false for "
              "background papers (methods, physiology, other populations) and true for any study that "
              "belongs. Then set `reviewed: true`.")
    write_yaml(path, {"how_to": how_to, "case": case.name, "topic": case.topic,
                      "reviews": [r.label for r in reviews], "reviewed": False, "papers": papers})
    return str(path), len(papers), sum(p["include"] for p in papers)


def share(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


@suite("search-recall")
async def search_recall(cases: list[EvalCase], services: Services) -> SuiteResult:
    """Live search for each case that has a reviewed recall file; recall at every stage + per source."""
    if services.discovery is None or services.screener is None:
        raise ValueError("the search-recall suite needs the discovery and screening services")
    s = get_settings()
    totals, details = Counter(), []
    found_by_source, unique_by_source, results_by_source = Counter(), Counter(), Counter()
    used = 0
    for case in cases:
        gold = load_recall_gold(case.name)
        if gold is None:
            continue
        used += 1
        # the stored plan keeps the queries fixed between runs; --replan tests the planner itself
        outcome = await services.discovery.search_only(case.topic, plan=None if services.replan else case.plan)
        if services.replan:
            details.append(f"{case.name} · queries: {outcome.plan.queries}")
        for r in outcome.agent_rounds:
            details.append(f"{case.name} · agent round {r['round']}: done={r['done']} "
                           f"new={r['new_queries']} (+{r.get('new_papers', 0)} papers) · gaps: {r['gaps']}")
        kept = prefilter(outcome.candidates, outcome.queries, keep=s.prefilter_keep)
        papers = [c.paper for c in kept]
        enough = max(s.fulltext_shortlist, s.select_max) if s.fulltext_enabled else s.select_max   # as research does
        screenings, screened = await screen_adaptive(services.screener, outcome.plan.research_brief, papers,
                                                     first=s.screen_first, step=s.screen_step, enough=enough)
        results_by_source.update(r.source for r in outcome.records)
        totals.update(gold=len(gold), candidates=len(outcome.candidates), source_errors=len(outcome.errors))
        details += [f"{case.name} · source error: {e}" for e in outcome.errors[:5]]
        for g in gold:
            hit = next((c for c in outcome.candidates if same_paper(g, c.paper)), None)
            label = f"{case.name} · {g.get('year')} {g['title'][:85]}"
            if hit is None:
                details.append(f"LOST AT SEARCH · {label}")
                continue
            totals["found"] += 1
            sources = {f.source for f in hit.found_by}
            found_by_source.update(sources)
            if len(sources) == 1:
                unique_by_source.update(sources)
            index = next((i for i, c in enumerate(kept) if c is hit), None)
            if index is None:
                details.append(f"LOST AT PREFILTER · {label} (found by {', '.join(sorted(sources))})")
                continue
            totals["prefiltered"] += 1
            if index >= screened:
                details.append(f"NOT SCREENED (adaptive stop) · {label} (prefilter rank {index + 1})")
                continue
            totals["screened"] += 1
            verdict = screenings.get(index)
            if verdict is None:
                details.append(f"SCREENING FAILED · {label}")
            elif verdict.relevance >= MIN_RELEVANCE:
                totals["relevant"] += 1
            else:
                details.append(f"SCREENED OUT · {label} → {verdict.relevance}/10: {verdict.reason}")
    if not used:
        raise ValueError("no case has a recall file: run `python -m app.evals prefill recall <case>` first")
    g = totals["gold"]
    metrics = {"gold_size": float(g), "candidates": float(totals["candidates"]),
               "recall_found": share(totals["found"], g), "recall_prefiltered": share(totals["prefiltered"], g),
               "recall_screened": share(totals["screened"], g), "recall_relevant": share(totals["relevant"], g),
               "source_errors": float(totals["source_errors"])}
    for source in sorted(set(results_by_source) | set(found_by_source)):
        metrics[f"results_{source}"] = float(results_by_source[source])         # raw results returned
        metrics[f"gold_found_by_{source}"] = float(found_by_source[source])     # gold papers it found
        metrics[f"gold_only_{source}"] = float(unique_by_source[source])        # ...that no other source found
    return metrics, details
