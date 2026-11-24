"""
Human labels for checking the judge (step 6.3).

    items file   every claim a grounding run judged (evals/results/<run>-grounding-items.yaml)
    label file   a sample of those items for YOU to label (evals/gold/judge_check.yaml, in git):
                 each item shows the claim, quote, paragraph and the judge's verdict; you fill in `human:`

The label file is never overwritten by accident: it holds your work.
"""

import random
from pathlib import Path

import yaml

from app.config import ROOT_DIR

GOLD_DIR = ROOT_DIR / "evals" / "gold"
JUDGE_CHECK = GOLD_DIR / "judge_check.yaml"
APPRAISAL = GOLD_DIR / "appraisal.yaml"
VERDICTS = ("supports", "partly", "does not support", "other study", "opposite")
FACT_VALUES = ("yes", "no", "not reported", "not applicable")
SEX_VALUES = ("male", "female", "mixed", "not reported", "not applicable")
DURATION_VALUES = ("single session", "under 3 months", "3 months or more", "not reported", "not applicable")


def write_yaml(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")
    return path


def read_yaml(path: Path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def make_label_file(items_path: Path, out: Path = JUDGE_CHECK, sample: int = 30, seed: int = 0,
                    force: bool = False) -> tuple[Path, int]:
    """Pick items to label: every one the judge did NOT call "supports" (the interesting ones), plus a
    random sample of "supports" items up to `sample`. Returns (path, number of items)."""
    if out.exists() and not force:
        raise FileExistsError(f"{out} already exists and may hold your labels (use --force to replace it)")
    items = read_yaml(items_path) or []
    doubtful = [i for i in items if i["judge"] != "supports"]
    supported = [i for i in items if i["judge"] == "supports"]
    random.Random(seed).shuffle(supported)
    picked = doubtful[:sample] + supported[:max(0, sample - len(doubtful))]
    for item in picked:
        item["human"] = None
    header = {"how_to": "For each item, read the paragraph and set `human:` to one of: " + ", ".join(VERDICTS)
              + ". Judge the CLAIM against the PARAGRAPH, not against what you know.", "items": picked}
    write_yaml(out, header)
    return out, len(picked)


def load_labels(path: Path = JUDGE_CHECK) -> list[dict]:
    """Labelled items; clear errors for a missing file, unfinished labels or a typo in a verdict."""
    if not path.exists():
        raise FileNotFoundError(f"no label file yet: run `python -m app.evals label <items file>` first ({path})")
    items = (read_yaml(path) or {}).get("items", [])
    missing = [i["id"] for i in items if not i.get("human")]
    if missing:
        raise ValueError(f"{len(missing)} of {len(items)} items still have no `human:` value (e.g. {missing[0]})")
    wrong = [(i["id"], i["human"]) for i in items if i["human"] not in VERDICTS]
    if wrong:
        raise ValueError(f"unknown verdict {wrong[0][1]!r} for {wrong[0][0]} (use one of: {', '.join(VERDICTS)})")
    return items


def _fact_value(value):
    """YAML reads an unquoted yes/no as true/false: turn them back into words."""
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return value


def load_appraisal_gold(path: Path = APPRAISAL) -> list[dict]:
    """Reviewed papers of the appraisal label file, with every value checked and normalised."""
    if not path.exists():
        raise FileNotFoundError(f"no appraisal labels yet: run `python -m app.evals prefill appraisal` first ({path})")
    papers = [p for p in (read_yaml(path) or {}).get("papers", []) if p.get("reviewed") is True]
    if not papers:
        raise ValueError(f"no paper in {path} is marked `reviewed: true` yet")
    for p in papers:
        for name, fact in p["facts"].items():
            gold = _fact_value(fact.get("gold"))
            allowed = {"sex": SEX_VALUES, "duration": DURATION_VALUES}.get(name, FACT_VALUES)
            if name == "sample_size":
                if gold is not None and not isinstance(gold, int):
                    raise ValueError(f"{p['id']} · sample_size: {gold!r} is not a number (or null)")
            elif gold not in allowed:
                raise ValueError(f"{p['id']} · {name}: unknown value {gold!r} (use one of: {', '.join(allowed)})")
            fact["gold"] = gold
        p["facts"] = {name: fact["gold"] for name, fact in p["facts"].items()}
    return papers


# ---------- step 6.5: screening and search-recall gold ----------

SCREENING = GOLD_DIR / "screening.yaml"
BUCKETS = ("relevant", "partly", "not relevant")


def recall_path(case: str) -> Path:
    return GOLD_DIR / f"recall_{case}.yaml"


def load_screening_gold(path: Path = SCREENING) -> dict[str, dict[int, dict]]:
    """{case: {candidate key: {"gold": bucket, "study_type": str | None}}} for reviewed cases only."""
    if not path.exists():
        raise FileNotFoundError(f"no screening labels yet: run `python -m app.evals prefill screening` first ({path})")
    gold: dict[str, dict[int, dict]] = {}
    for block in (read_yaml(path) or {}).get("cases", []):
        if block.get("reviewed") is not True:
            continue
        for p in block["papers"]:
            if p.get("gold") not in BUCKETS:
                raise ValueError(f"{block['case']} · {p['key']}: unknown bucket {p.get('gold')!r} "
                                 f"(use one of: {', '.join(BUCKETS)})")
            gold.setdefault(block["case"], {})[p["key"]] = {"gold": p["gold"], "study_type": p.get("gold_study_type")}
    if not gold:
        raise ValueError(f"no case in {path} is marked `reviewed: true` yet")
    return gold


def load_recall_gold(case: str) -> list[dict] | None:
    """The confirmed gold papers (include: true) of a case, or None if the case has no recall file."""
    path = recall_path(case)
    if not path.exists():
        return None
    data = read_yaml(path) or {}
    if data.get("reviewed") is not True:
        raise ValueError(f"{path} is not marked `reviewed: true` yet")
    return [p for p in data.get("papers", []) if p.get("include") is True]
