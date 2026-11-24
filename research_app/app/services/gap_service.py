"""
Shared limitations and research gaps (step 5.4).

Strict division of labour:
  code COUNTS    designs, sexes, sample sizes, durations, how often each quality item is present
  code FINDS     gaps with explicit rules (policy as data: GAP_RULES) and groups the authors' own
                 stated limitations by theme (THEMES)
  LLM WRITES     a short summary + suggested next studies from those facts ONLY
  code CHECKS    every number in the write-up must appear in the counted facts; if the check fails or
                 the call fails, the report still shows the code's bullet points
"""

import statistics
from collections import Counter
from dataclasses import dataclass, field

from app.common.logging import get_logger
from app.domain.errors import LlmError
from app.domain.schemas import AnalyzedPaper, EvidenceGaps, GapWriteUp, SharedLimitation
from app.integrations.llm.gateway import LlmGateway
from app.services.grounding import numbers_missing

log = get_logger(__name__)


@dataclass
class PaperFacts:
    """What step 5.4 needs from each paper (screening + analysis + summary)."""
    label: str                      # in-text citation, e.g. "(Smith et al., 2020)"
    study_type: str
    based_on: str                   # "full text" or "abstract"
    sample_size: int | None
    sex: str
    duration: str
    facts: dict[str, str]           # control_group -> yes / no / not reported / not applicable ...
    limitations: list[str] = field(default_factory=list)      # stated by the authors (4.5 summary)


def paper_facts(label: str, study_type: str, analyzed: AnalyzedPaper, limitations: list[str]) -> PaperFacts:
    a = analyzed.analysis.appraisal
    facts = {name: getattr(a, name).value for name in ("control_group", "randomized", "blinded",
             "statistics_reported", "effect_sizes", "conflicts_declared", "data_available")}
    return PaperFacts(label=label, study_type=study_type, based_on=analyzed.based_on, sample_size=a.sample_size,
                      sex=a.sex, duration=a.duration, facts=facts, limitations=list(limitations))


@dataclass
class Profile:
    """The counts every gap rule looks at."""
    n_papers: int
    designs: Counter
    sexes: Counter
    durations: Counter
    sample_sizes: list[int]
    fact_yes: Counter               # papers with "yes", per fact
    fact_applicable: Counter        # papers the fact applies to, per fact
    abstract_only: int
    human_studies: int              # papers with participants (sex not "not applicable")


def profile(papers: list[PaperFacts]) -> Profile:
    fact_yes, fact_applicable = Counter(), Counter()
    for p in papers:
        for name, value in p.facts.items():
            if value != "not applicable":
                fact_applicable[name] += 1
                fact_yes[name] += value == "yes"
    return Profile(
        n_papers=len(papers), designs=Counter(p.study_type for p in papers),
        sexes=Counter(p.sex for p in papers), durations=Counter(p.duration for p in papers),
        sample_sizes=sorted(p.sample_size for p in papers if p.sample_size is not None),
        fact_yes=fact_yes, fact_applicable=fact_applicable,
        abstract_only=sum(p.based_on != "full text" for p in papers),
        human_studies=sum(p.sex != "not applicable" for p in papers))


# ---------- gap rules: each looks at the counts and returns a sentence, or None (policy as data) ----------

def _mostly_people(pr: Profile) -> bool:
    """Participant-based rules (trials, sex, sample size) only make sense if most studies have participants."""
    return pr.human_studies * 2 >= pr.n_papers


def _no_trials(pr: Profile) -> str | None:
    if _mostly_people(pr) and pr.designs["randomized_trial"] == 0:
        return "No randomised controlled trial among the selected papers."
    return None


def _no_reviews(pr: Profile) -> str | None:
    if pr.designs["meta_analysis"] + pr.designs["systematic_review"] == 0:
        return "No systematic review or meta-analysis pools these results."
    return None


def _small_samples(pr: Profile) -> str | None:
    if _mostly_people(pr) and pr.sample_sizes and statistics.median(pr.sample_sizes) < 20:
        tiny = sum(n < 10 for n in pr.sample_sizes)
        return (f"Small samples: median n = {statistics.median(pr.sample_sizes):g} "
                f"(range {pr.sample_sizes[0]}–{pr.sample_sizes[-1]}); {tiny} of {len(pr.sample_sizes)} "
                f"studies had fewer than 10 participants.")
    return None


def _one_sex(pr: Profile) -> str | None:
    reported = pr.n_papers - pr.sexes["not reported"] - pr.sexes["not applicable"]
    single = pr.sexes["male"] + pr.sexes["female"]
    if reported and single * 2 >= reported:
        return (f"{single} of {reported} studies that report sex included only one sex "
                f"({pr.sexes['female']} women only, {pr.sexes['male']} men only).")
    return None


def _sex_unreported(pr: Profile) -> str | None:
    if pr.sexes["not reported"] >= 2:
        return f"{pr.sexes['not reported']} studies do not report the participants' sex."
    return None


def _short(pr: Profile) -> str | None:
    known = pr.n_papers - pr.durations["not reported"] - pr.durations["not applicable"]
    long_term = pr.durations["3 months or more"]
    if known >= 3 and long_term <= 1:
        return (f"Mostly short studies: {pr.durations['single session']} single-session and "
                f"{pr.durations['under 3 months']} under 3 months; {long_term} lasted 3 months or more.")
    return None


def _rare(fact: str, none_text: str, some_text: str):
    """A quality item present in a third of the studies or fewer."""
    def rule(pr: Profile) -> str | None:
        applicable, yes = pr.fact_applicable[fact], pr.fact_yes[fact]
        if applicable < 3 or yes * 3 > applicable:
            return None
        return none_text if yes == 0 else f"{some_text} in only {yes} of {applicable} studies."
    return rule


def _abstracts(pr: Profile) -> str | None:
    if pr.abstract_only and pr.abstract_only * 3 >= pr.n_papers:
        return (f"{pr.abstract_only} of {pr.n_papers} papers could only be judged from their abstracts "
                f"(no free full text), so their methods are less certain.")
    return None


GAP_RULES = [
    _no_trials, _no_reviews, _small_samples, _one_sex, _sex_unreported, _short,
    _rare("blinded", "No study reports blinding.", "Blinding reported"),
    _rare("effect_sizes", "No study reports effect sizes.", "Effect sizes reported"),
    _rare("data_available", "No study shares its data or code.", "Data or code shared"),
    _rare("control_group", "No study has a control group.", "A control group"),
]
# Notes are about what THIS APP could read, not gaps in the research itself. They are shown to the
# reader but NOT given to the write-up (otherwise it suggests "studies" to fix our own blind spots).
NOTE_RULES = [_abstracts]


def find_gaps(pr: Profile) -> list[str]:
    return [text for rule in GAP_RULES if (text := rule(pr))]


def find_notes(pr: Profile) -> list[str]:
    return [text for rule in NOTE_RULES if (text := rule(pr))]


# ---------- shared limitations: the authors' own words, grouped by theme (code) ----------

THEMES: dict[str, tuple[str, ...]] = {
    "small sample": ("sample size", "small sample", "number of participants", "few participants", "small number"),
    "one sex / narrow population": ("only men", "only women", "male", "female", "untrained", "trained",
                                    "young", "generali", "population"),
    "short duration / acute only": ("short", "acute", "long-term", "longer", "duration", "weeks"),
    "measurement method": ("emg", "electrode", "crosstalk", "ultrasound", "measure", "signal"),
    "not all muscles or heads studied": ("head", "other muscle", "only the", "not evaluated", "not assessed"),
}


def shared_limitations(papers: list[PaperFacts]) -> list[SharedLimitation]:
    """Themes that the authors of at least 2 papers admit, most common first."""
    by_theme: dict[str, list[str]] = {}
    for p in papers:
        text = " ".join(p.limitations).lower()
        for theme, words in THEMES.items():
            if any(w in text for w in words):
                by_theme.setdefault(theme, []).append(p.label)
    shared = [SharedLimitation(theme=t, papers=labels) for t, labels in by_theme.items() if len(labels) >= 2]
    return sorted(shared, key=lambda s: -len(s.papers))


# ---------- the LLM only writes it up ----------

WRITE_PROMPT = """You write the "gaps" part of a literature review.
Research brief: {brief}

These facts were COUNTED by code from the {n} selected papers. Use ONLY these facts and numbers;
do not add new gaps, numbers or claims.

Gaps found:
{gaps}

Limitations the authors themselves admit:
{limits}

Write:
- summary: 2-4 plain sentences on what this body of evidence is missing and why it matters.
- next_studies: up to 4 concrete studies that would fill the most important gaps (describe them in
  words; do not put numbers in them)."""


class GapService:
    def __init__(self, gateway: LlmGateway, max_tokens: int = 2000):
        self.gateway = gateway
        self.max_tokens = max_tokens

    async def find(self, brief: str, papers: list[PaperFacts]) -> EvidenceGaps:
        counts = profile(papers)
        gaps = find_gaps(counts)
        shared = shared_limitations(papers)
        stating = sum(bool(p.limitations) for p in papers)
        result = EvidenceGaps(n_papers=len(papers), gaps=gaps, notes=find_notes(counts), shared_limitations=shared,
                              papers_with_limitations=stating)
        if not gaps and not shared:
            return result
        limits = "\n".join(f"- {s.theme} ({len(s.papers)} papers)" for s in shared) or \
            f"- none shared by 2 or more papers ({stating} of {len(papers)} papers state any limitations)"
        prompt = WRITE_PROMPT.format(brief=brief, n=len(papers), limits=limits,
                                     gaps="\n".join(f"- {g}" for g in gaps) or "- none found")
        try:
            write_up, _ = await self.gateway.complete_structured(
                [{"role": "user", "content": prompt}], GapWriteUp, purpose="gaps", max_tokens=self.max_tokens)
        except LlmError as e:
            log.warning("gap write-up failed: %s", e)
            result.flags.append("no written summary (LLM call failed): the counted gaps are shown as they are")
            return result
        # the summary describes the evidence: every number must be a counted one, or it is dropped
        invented = numbers_missing([write_up.summary], prompt)
        if invented:
            result.flags.append(f"written summary dropped: it used numbers not in the counted facts "
                                f"({', '.join(invented)})")
            return result
        # suggested studies are proposals, not facts: drop only a suggestion that slips in an uncounted number
        studies = []
        for study in write_up.next_studies:
            if extra := numbers_missing([study], prompt):
                result.flags.append(f"suggested study dropped (uses numbers not in the counted facts: "
                                    f"{', '.join(extra)}): {study[:80]}")
            else:
                studies.append(study)
        result.write_up = GapWriteUp(summary=write_up.summary, next_studies=studies)
        return result
