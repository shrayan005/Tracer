"""
Appraisal suite (step 6.4): are the AI's quality-checklist facts TRUE? Measured against YOUR labels.

    prefill   evals/gold/appraisal.yaml: for each full-text paper, the AI's facts + quotes, plus HINTS (the
              sentences in the paper that decide each fact), with `gold:` pre-filled from the AI.
              You correct the wrong ones and mark the paper `reviewed: true`.
    suite     AI facts vs your gold: accuracy per fact, FALSE ✓ (raises trust wrongly), MISSED ✓ (lowers it
              unfairly), sample size, and whether the AI's errors change the TRUST LEVEL a reader sees.

Only full-text papers: from an abstract the AI cannot see blinding or conflict statements, so scoring
it against the full truth would measure our lack of access, not the AI.
"""

import re

from app.domain.schemas import Appraisal, Fact
from app.evals.cases import EvalCase
from app.evals.grounding import claims_for
from app.evals.labels import APPRAISAL, load_appraisal_gold, write_yaml
from app.evals.runner import Services, SuiteResult, suite
from app.services.quality import apply_quality

CHECKLIST = ("control_group", "randomized", "blinded", "statistics_reported", "effect_sizes",
             "conflicts_declared", "data_available")
FACTS = ("sample_size", "sex", "duration", *CHECKLIST)
NUMBER_WORDS = (r"\d{1,4}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
                r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty[- ]?\w*|thirty[- ]?\w*|forty[- ]?\w*")
HINTS = {
    # a number, then up to 4 describing words (commas allowed: "Nine young, novice, female participants")
    "sample_size": rf"\b({NUMBER_WORDS})\b(?:[\s,]+[a-z-]+){{0,4}}?[\s,]+(participants|subjects|men|women|"
                   rf"volunteers|patients|adults|students|individuals|athletes|pitchers|arms)\b",
    "sex": r"\b(men|women|males?|females?)\b",
    "duration": r"\b\d+\s*(weeks?|months?|days?|sessions?)\b|\bsingle (session|visit)\b",
    "control_group": r"\bcontrol(s| group| condition| arm)?\b|\bplacebo\b|\bsham\b|\bcomparison group\b|contralateral",
    "randomized": r"\brandom(ly|i[sz]ed|i[sz]ation)?\b",
    "blinded": r"\bblind(ed|ing)?\b|\bmask(ed|ing)\b",
    "statistics_reported": r"\bp\s*[<=>≤]\s*0?\.\d|\banova\b|\bt-test\b|confidence interval|\b95\s*%\s*ci\b|"
                           r"statistical(ly)? (analys|signific)",
    "effect_sizes": r"effect size|cohen'?s d|\bhedges\b|\beta[- ]?squared\b|\bη2|\bηp2|partial eta|\bES\s*=",
    "conflicts_declared": r"conflicts? of interest|competing interests?|declare[sd]? (that )?(there is |no )",
    "data_available": r"data (availability|are available|is available)|available (on|upon) (reasonable )?request|"
                      r"supplementary (data|material)|github|osf\.io|zenodo|figshare|repository",
}
SENTENCE = re.compile(r"(?<=[.!?])\s+")


def hints(sections, fact: str, limit: int = 2, max_len: int = 260) -> list[str]:
    """Up to `limit` sentences (with their section) that mention what decides this fact."""
    pattern = re.compile(HINTS[fact], re.IGNORECASE)
    found = []
    for kind, _title, text in sections:
        if kind in ("references", "front"):
            continue
        for sentence in SENTENCE.split(text.replace("\n", " ")):
            if pattern.search(sentence):
                s = sentence.strip()
                found.append(f"[{kind}] {s[:max_len]}{'…' if len(s) > max_len else ''}")
                if len(found) >= limit:
                    return found
    return found


def ai_facts(a: Appraisal) -> dict[str, tuple]:
    """fact -> (AI value, AI quote)"""
    out = {"sample_size": (a.sample_size, a.sample_size_quote), "sex": (a.sex, None), "duration": (a.duration, None)}
    out |= {f: (getattr(a, f).value, getattr(a, f).quote) for f in CHECKLIST}
    return out


async def prefill(cases: list[EvalCase], services: Services, force: bool = False) -> tuple[str, int]:
    """Write the label file for the full-text papers of the cases. Returns (path, number of papers)."""
    if APPRAISAL.exists() and not force:
        raise FileExistsError(f"{APPRAISAL} already exists and may hold your labels (use --force to replace it)")
    papers = []
    for case in cases:
        analyses, _ = await claims_for(case, services)
        for p in case.papers:
            if not p.sections or p.key not in analyses:
                continue
            facts = {}
            for name, (value, quote) in ai_facts(analyses[p.key].analysis.appraisal).items():
                facts[name] = {"ai": value, "ai_quote": quote, "hints": hints(p.sections, name), "gold": value}
            link = f"https://doi.org/{p.paper.doi}" if p.paper.doi else p.paper.landing_url
            papers.append({"id": f"{case.name}-{p.key}", "paper": f"{p.label} {p.paper.title}", "link": link,
                           "reviewed": False, "facts": facts})
    header = {"how_to": "For each paper: check every `gold:` value (pre-filled with the AI's answer) and correct "
                        "the wrong ones, then set `reviewed: true`. Facts: yes / no / not reported / not applicable. "
                        "sex: male / female / mixed / not reported / not applicable. duration: single session / "
                        "under 3 months / 3 months or more / not reported / not applicable. sample_size: a number "
                        "or null.", "papers": papers}
    write_yaml(APPRAISAL, header)
    return str(APPRAISAL), len(papers)


# ---------- comparing ----------

NEGATIVE = {"no", "not reported"}           # they score the same in the quality check (step 5.2)


def same(fact: str, ai, gold) -> bool:
    if fact == "sample_size":
        return ai == gold or (ai is not None and gold is not None and abs(ai - gold) <= 1)
    return ai == gold or (ai in NEGATIVE and gold in NEGATIVE)


def gold_appraisal(gold: dict) -> Appraisal:
    return Appraisal(sample_size=gold["sample_size"], sex=gold["sex"], duration=gold["duration"],
                     **{f: Fact(value=gold[f]) for f in CHECKLIST})


@suite("appraisal")
async def appraisal(cases: list[EvalCase], services: Services) -> SuiteResult:
    gold_papers = {g["id"]: g for g in load_appraisal_gold()}           # only reviewed papers
    checked = correct = false_yes = missed_yes = ai_yes = gold_yes = 0
    per_fact = {f: [0, 0] for f in FACTS}                               # fact -> [correct, checked]
    trust_same = trust_total = 0
    details = []
    for case in cases:
        wanted = [p for p in case.papers if f"{case.name}-{p.key}" in gold_papers]
        if not wanted:
            continue
        # analyse only the gold papers: half the calls, so the free tier doesn't push some to another model
        analyses, warnings = await claims_for(case.model_copy(update={"papers": wanted}), services)
        details += [f"{case.name}: {w}" for w in warnings]
        for p in wanted:
            if p.key not in analyses:
                details.append(f"{case.name} · {p.label}: no analysis")
                continue
            gold = gold_papers[f"{case.name}-{p.key}"]["facts"]
            ai = analyses[p.key].analysis.appraisal
            for name, (value, _quote) in ai_facts(ai).items():
                ok = same(name, value, gold[name])
                checked += 1
                correct += ok
                per_fact[name][0] += ok
                per_fact[name][1] += 1
                if name in CHECKLIST:
                    ai_yes += value == "yes"
                    gold_yes += gold[name] == "yes"
                    false_yes += value == "yes" and gold[name] != "yes"
                    missed_yes += gold[name] == "yes" and value != "yes"
                if not ok:
                    details.append(f"{case.name} · {p.label} · {name}: AI {value}, you {gold[name]}")
            ai_trust = apply_quality(p.base_trust, ai, case.field_family, "full text").level
            gold_trust = apply_quality(p.base_trust, gold_appraisal(gold), case.field_family, "full text").level
            trust_total += 1
            trust_same += ai_trust == gold_trust
            if ai_trust != gold_trust:
                details.append(f"{case.name} · {p.label} · TRUST: AI's facts give {ai_trust}, yours give {gold_trust}")

    rate = (lambda n, d: n / d if d else 0.0)
    metrics = {"papers": float(trust_total), "facts": float(checked), "fact_accuracy": rate(correct, checked),
               "false_yes": float(false_yes), "false_yes_rate": rate(false_yes, ai_yes),
               "missed_yes": float(missed_yes), "missed_yes_rate": rate(missed_yes, gold_yes),
               "sample_size_accuracy": rate(*per_fact["sample_size"]), "trust_agreement": rate(trust_same, trust_total)}
    metrics |= {f"accuracy_{f}": rate(*per_fact[f]) for f in FACTS if f != "sample_size" and per_fact[f][1]}
    return metrics, details
