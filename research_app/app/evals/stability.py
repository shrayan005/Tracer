"""
Stability suite (step 6.2): analyse every paper N times on the SAME frozen input and measure what changes.
No gold labels needed: the runs are compared with each other.

  facts       sample size, sex, duration + the 7 checklist items: same value in every run?
  trust       the quality-adjusted trust level per run: does it move?
  claims      same number of claims? do the claims match between runs (direction + similar words)?
  connections the connection step run N times on the SAME claims (run 1's), so its own instability is
              measured separately from the claims': how much do the grouped pairs overlap?
"""

from collections import Counter
from itertools import combinations

from app.domain.schemas import AnalyzedPaper
from app.evals.cases import EvalCase
from app.evals.runner import Services, SuiteResult, suite
from app.services.prefilter import keywords
from app.services.quality import apply_quality

FACTS = ("sample_size", "sex", "duration", "control_group", "randomized", "blinded", "statistics_reported",
         "effect_sizes", "conflicts_declared", "data_available")


def fact_values(a: AnalyzedPaper) -> dict[str, str]:
    ap = a.analysis.appraisal
    return {f: str(getattr(ap, f)) if f in ("sample_size", "sex", "duration") else getattr(ap, f).value
            for f in FACTS}


def claims_match(x, y) -> bool:
    """Same result in two runs: same direction and mostly the same words for factor + outcome."""
    wx, wy = keywords([x.factor, x.outcome]), keywords([y.factor, y.outcome])
    return x.direction == y.direction and bool(wx | wy) and len(wx & wy) / len(wx | wy) >= 0.4


def claim_overlap(a: AnalyzedPaper, b: AnalyzedPaper) -> float:
    """Share of claims in each run that have a matching claim in the other run (averaged both ways)."""
    ca, cb = a.analysis.claims, b.analysis.claims
    if not ca or not cb:
        return 0.0
    ab = sum(any(claims_match(x, y) for y in cb) for x in ca) / len(ca)
    ba = sum(any(claims_match(y, x) for x in ca) for y in cb) / len(cb)
    return (ab + ba) / 2


def connection_pairs(connections) -> set[frozenset]:
    """Every pair of (paper, claim) that a run grouped together: the 'shape' of its connections."""
    pairs = set()
    for c in connections:
        members = [(m.paper, m.claim) for m in c.members]
        pairs |= {frozenset(p) for p in combinations(members, 2)}
    return pairs


def jaccard(a: set, b: set) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


@suite("stability")
async def stability(cases: list[EvalCase], services: Services) -> SuiteResult:
    runs = services.runs
    facts_total = facts_stable = trust_total = trust_stable = count_stable = 0
    overlaps, flips, details = [], Counter(), []
    connection_scores, group_counts = [], []
    for case in cases:
        papers = {p.key: p.paper for p in case.papers}
        texts = {p.key: t for p in case.papers if (t := p.text()) is not None}
        by_run = []
        for _ in range(runs):                               # one run at a time: gentle on rate limits
            results, warnings = await services.analysis.analyze(papers, texts)
            by_run.append(results)
            details += [f"{case.name}: {w}" for w in warnings]
        for p in case.papers:
            got = [r[p.key] for r in by_run if p.key in r]
            if len(got) < 2:
                continue
            values = [fact_values(a) for a in got]
            for f in FACTS:
                facts_total += 1
                seen = [v[f] for v in values]
                if len(set(seen)) == 1:
                    facts_stable += 1
                else:
                    flips[f] += 1
                    details.append(f"{case.name} · {p.label} · {f}: {' / '.join(seen)}")
            levels = [apply_quality(p.base_trust, a.analysis.appraisal, case.field_family, a.based_on).level
                      for a in got]
            trust_total += 1
            if len(set(levels)) == 1:
                trust_stable += 1
            else:
                details.append(f"{case.name} · {p.label} · trust level: {' / '.join(map(str, levels))}")
            counts = [len(a.analysis.claims) for a in got]
            count_stable += len(set(counts)) == 1
            overlaps += [claim_overlap(x, y) for x, y in combinations(got, 2)]

        if services.connection is not None:                 # connections: same claims (run 1), N times
            first = by_run[0]
            inputs = [(p.key, p.label, p.study_type, first[p.key]) for p in case.papers if p.key in first]
            shapes = []
            for _ in range(runs):
                connections, _ = await services.connection.connect(case.brief, inputs)
                shapes.append(connection_pairs(connections))
                group_counts.append(len(connections))
            connection_scores += [jaccard(a, b) for a, b in combinations(shapes, 2)]
            details.append(f"{case.name} · connection groups per run: "
                           f"{' / '.join(map(str, group_counts[-runs:]))}")

    metrics = {
        "facts_identical_share": facts_stable / facts_total if facts_total else 0.0,
        "trust_identical_share": trust_stable / trust_total if trust_total else 0.0,
        "claim_count_identical_share": count_stable / trust_total if trust_total else 0.0,
        "claim_overlap_mean": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        "papers": float(trust_total),
        "runs": float(runs),
    }
    if connection_scores:
        metrics["connection_overlap_mean"] = sum(connection_scores) / len(connection_scores)
        metrics["connection_groups_mean"] = sum(group_counts) / len(group_counts)
    details.insert(0, "most unstable facts: " + (", ".join(f"{f} ({n})" for f, n in flips.most_common()) or "none"))
    return metrics, details
