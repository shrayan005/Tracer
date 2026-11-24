"""
Connections eval (step 6.6): does the connection step link the right claims, with the right stances?

Claims change with every analysis, so the gold file FREEZES them: per case, every paper's claims
(id, factor, outcome, direction, population) plus the gold groups ("these claims answer the same
question", each with a stance). The suite re-runs the connection step on exactly those claims.

    prefill   evals/gold/connections.yaml: the stored analyses' claims + one live connection run as a
              first draft of the groups, for YOU to correct (remove wrong members, add missed ones)
    suite     connects the frozen claims `--runs` times and scores each run:
                pair precision / recall   claims from different papers put in the same group
                groups found              gold groups matched by an AI group (Jaccard >= 0.5)
                relation / stance         agree-contradict-mixed (code computes it) and per-claim stances
"""

from itertools import combinations

from app.domain.schemas import AnalyzedPaper, Appraisal, Claim, Connection, Fact, PaperAnalysis
from app.evals.cases import EvalCase
from app.evals.labels import GOLD_DIR, read_yaml, write_yaml
from app.evals.runner import Services, SuiteResult, suite
from app.repositories.analysis_repository import AnalysisRepository
from app.repositories.session import session_scope
from app.services.connection_service import relation_of

CONNECTIONS = GOLD_DIR / "connections.yaml"
STANCES = ("supports", "opposes", "no effect")
MATCH = 0.5                       # Jaccard of members needed to call an AI group "the same" as a gold group
CLAIM_FIELDS = ("factor", "outcome", "direction", "population", "magnitude")


# ---------- frozen claims <-> connection-step input ----------

def as_input(paper: dict) -> tuple[int, str, str, AnalyzedPaper]:
    """A frozen paper of the gold file -> the (key, label, study type, analysis) the connection step takes."""
    facts = {f: Fact(value="not reported") for f in ("control_group", "randomized", "blinded", "statistics_reported",
                                                     "effect_sizes", "conflicts_declared", "data_available")}
    appraisal = Appraisal(sample_size=paper.get("n"), sex=paper["sex"], duration=paper["duration"], **facts)
    claims = [Claim(quote="", **{k: c.get(k) for k in CLAIM_FIELDS}) for c in paper["claims"]]
    return paper["key"], paper["label"], paper["study_type"], AnalyzedPaper(
        analysis=PaperAnalysis(claims=claims, appraisal=appraisal), based_on="abstract")


def claim_ids(connection: Connection, keys: list[int]) -> dict[str, str]:
    """{claim id like 'P2.C1': stance} for one checked connection (ids by paper position, as in the prompt)."""
    position = {key: n for n, key in enumerate(keys, start=1)}
    return {f"P{position[m.paper]}.C{m.claim + 1}": m.stance for m in connection.members if m.paper in position}


def pairs(groups: list[dict[str, str]]) -> set[frozenset]:
    """Every pair of claims from DIFFERENT papers that share a group."""
    out = set()
    for g in groups:
        for a, b in combinations(sorted(g), 2):
            if a.split(".")[0] != b.split(".")[0]:
                out.add(frozenset((a, b)))
    return out


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


# ---------- the gold file ----------

async def prefill_connections(cases: list[EvalCase], services: Services, force: bool = False) -> tuple[str, int]:
    if CONNECTIONS.exists() and not force:
        raise FileExistsError(f"{CONNECTIONS} already exists and may hold your labels (use --force to replace it)")
    if services.connection is None:
        raise ValueError("prefill connections needs the connection service")
    blocks, total = [], 0
    for case in cases:
        async with session_scope() as session:                 # the analyses the reports use now
            stored = await AnalysisRepository(session).get_many([p.key for p in case.papers])
        papers = []
        for p in case.papers:
            if p.key not in stored:
                continue
            a = AnalyzedPaper.model_validate(stored[p.key].analysis).analysis
            papers.append({"key": p.key, "label": p.label, "study_type": p.study_type, "n": a.appraisal.sample_size,
                           "sex": a.appraisal.sex, "duration": a.appraisal.duration,
                           "claims": [{"id": f"P{len(papers) + 1}.C{m}", **{k: getattr(c, k) for k in CLAIM_FIELDS}}
                                      for m, c in enumerate(a.claims, start=1)]})
        if len(papers) < 2:
            continue
        inputs = [as_input(p) for p in papers]
        connections, _ = await services.connection.connect(case.brief, inputs)
        keys = [p["key"] for p in papers]
        groups = []
        for c in connections:
            members = claim_ids(c, keys)
            groups.append({"question": c.question, "members": [{"claim": k, "stance": s} for k, s in members.items()]})
        total += len(groups)
        blocks.append({"case": case.name, "brief": case.brief, "reviewed": False, "papers": papers, "groups": groups})
    how_to = ("Each group = claims from different papers that answer the SAME question. Remove members that "
              "don't belong, add missed claims (ids like P3.C2), fix stances (" + ", ".join(STANCES) + "), "
              "merge or split groups. Then set `reviewed: true` for the case.")
    write_yaml(CONNECTIONS, {"how_to": how_to, "cases": blocks})
    return str(CONNECTIONS), total


def load_connections_gold(path=CONNECTIONS) -> dict[str, dict]:
    """{case: block} for reviewed cases; every group checked (known ids, stances, 2+ papers)."""
    if not path.exists():
        raise FileNotFoundError(f"no connection labels yet: run `python -m app.evals prefill connections` ({path})")
    gold = {}
    for block in (read_yaml(path) or {}).get("cases", []):
        if block.get("reviewed") is not True:
            continue
        known = {c["id"] for p in block["papers"] for c in p["claims"]}
        for g in block["groups"]:
            for m in g["members"]:
                if m["claim"] not in known:
                    raise ValueError(f"{block['case']}: unknown claim id {m['claim']!r} in '{g['question'][:50]}'")
                if m["stance"] not in STANCES:
                    raise ValueError(f"{block['case']}: unknown stance {m['stance']!r} (use: {', '.join(STANCES)})")
            if len({m["claim"].split(".")[0] for m in g["members"]}) < 2:
                raise ValueError(f"{block['case']}: group '{g['question'][:50]}' needs claims from 2+ papers")
        gold[block["case"]] = block
    if not gold:
        raise ValueError(f"no case in {path} is marked `reviewed: true` yet")
    return gold


# ---------- the suite ----------

@suite("connections")
async def connections(cases: list[EvalCase], services: Services) -> SuiteResult:
    if services.connection is None:
        raise ValueError("the connections suite needs the connection service")
    gold = load_connections_gold()
    scores = {k: [] for k in ("pair_precision", "pair_recall", "groups_found", "relation_accuracy",
                              "stance_accuracy", "ai_groups")}
    details = []
    for case in cases:
        block = gold.get(case.name)
        if block is None:
            continue
        inputs = [as_input(p) for p in block["papers"]]
        keys = [p["key"] for p in block["papers"]]
        gold_groups = [{m["claim"]: m["stance"] for m in g["members"]} for g in block["groups"]]
        gold_pairs = pairs(gold_groups)
        for run in range(1, services.runs + 1):
            found, flags = await services.connection.connect(block["brief"], inputs)
            if run == 1:
                details += [f"CHECK · {case.name} · {f}" for f in flags]      # what the code's checks removed
            ai_groups = [claim_ids(c, keys) for c in found]
            ai_pairs = pairs(ai_groups)
            right = gold_pairs & ai_pairs
            scores["pair_precision"].append(len(right) / len(ai_pairs) if ai_pairs else 1.0)
            scores["pair_recall"].append(len(right) / len(gold_pairs) if gold_pairs else 1.0)
            scores["ai_groups"].append(len(ai_groups))
            matched = relations = stances_right = stances_total = 0
            for g, question in zip(gold_groups, (g["question"] for g in block["groups"])):
                best = max(ai_groups, key=lambda a: jaccard(set(a), set(g)), default=None)
                if best is None or jaccard(set(best), set(g)) < MATCH:
                    if run == 1:
                        details.append(f"MISSED GROUP · {case.name} · {question[:90]}")
                    continue
                matched += 1
                relations += relation_of(list(best.values())) == relation_of(list(g.values()))
                shared = set(best) & set(g)
                stances_total += len(shared)
                stances_right += sum(best[c] == g[c] for c in shared)
            scores["groups_found"].append(matched / len(gold_groups) if gold_groups else 1.0)
            if matched:
                scores["relation_accuracy"].append(relations / matched)
            if stances_total:
                scores["stance_accuracy"].append(stances_right / stances_total)
            if run == 1:
                details += [f"FALSE LINK · {case.name} · {' ↔ '.join(sorted(p))}" for p in sorted(ai_pairs - gold_pairs, key=sorted)][:10]
                details += [f"MISSED LINK · {case.name} · {' ↔ '.join(sorted(p))}" for p in sorted(gold_pairs - ai_pairs, key=sorted)][:10]
    if not scores["pair_recall"]:
        raise ValueError("no reviewed connection labels for these cases")
    metrics = {k: sum(v) / len(v) for k, v in scores.items() if v}
    metrics["runs"] = float(services.runs)
    return metrics, details
