"""
Scorecards: the result of one eval run, saved as JSON (for comparing) + Markdown (for reading).

A scorecard records WHAT was measured (metrics), on WHICH cases, with WHICH settings (models, routes,
prompt version), and what it COST (from llm_calls, tagged with the run id). Comparing two scorecards
shows exactly what a change did.
"""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import ROOT_DIR

RESULTS_DIR = ROOT_DIR / "evals" / "results"


class StageCost(BaseModel):
    calls: int
    tokens: int
    seconds: float


class Scorecard(BaseModel):
    suite: str
    run_id: int
    started_at: datetime
    seconds: float
    cases: list[str]
    settings: dict[str, str]                        # models, routes, prompt versions at run time
    metrics: dict[str, float] = Field(default_factory=dict)
    details: list[str] = Field(default_factory=list)  # human-readable findings (what flipped, what failed)
    cost: dict[str, StageCost] = Field(default_factory=dict)   # per LLM purpose

    @property
    def total_tokens(self) -> int:
        return sum(c.tokens for c in self.cost.values())


def save_scorecard(card: Scorecard) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{card.started_at:%Y%m%d-%H%M%S}-{card.suite}"
    json_path, md_path = RESULTS_DIR / f"{stem}.json", RESULTS_DIR / f"{stem}.md"
    json_path.write_text(card.model_dump_json(indent=1), encoding="utf-8")
    md_path.write_text(render_scorecard(card), encoding="utf-8")
    return json_path, md_path


def load_scorecard(path: str | Path) -> Scorecard:
    return Scorecard.model_validate_json(Path(path).read_text(encoding="utf-8"))


def render_scorecard(card: Scorecard) -> str:
    lines = [f"# Eval: {card.suite}", "",
             f"Run {card.run_id} · {card.started_at:%Y-%m-%d %H:%M} · {card.seconds:.0f}s · "
             f"cases: {', '.join(card.cases)} · {card.total_tokens:,} tokens", "",
             "## Metrics", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {name} | {_fmt(value)} |" for name, value in card.metrics.items()]
    lines += ["", "## Cost", "", "| LLM purpose · provider | Calls | Tokens | Seconds |", "|---|---|---|---|"]
    lines += [f"| {p} | {c.calls} | {c.tokens:,} | {c.seconds:.0f} |" for p, c in card.cost.items()]
    lines += ["", "## Settings", ""] + [f"- {k}: `{v}`" for k, v in card.settings.items()]
    if card.details:
        lines += ["", "## Details", ""] + [f"- {d}" for d in card.details]
    return "\n".join(lines) + "\n"


def compare(old: Scorecard, new: Scorecard) -> str:
    """A table of every metric in either card, with the change. Settings that differ are listed too."""
    names = list(dict.fromkeys([*old.metrics, *new.metrics]))
    lines = [f"Comparing {old.suite} run {old.run_id} ({old.started_at:%Y-%m-%d %H:%M}) → run {new.run_id} "
             f"({new.started_at:%Y-%m-%d %H:%M})", "", "| Metric | Before | After | Change |", "|---|---|---|---|"]
    for name in names:
        a, b = old.metrics.get(name), new.metrics.get(name)
        change = "" if a is None or b is None else f"{b - a:+.3g}"
        lines.append(f"| {name} | {_fmt(a)} | {_fmt(b)} | {change} |")
    lines.append(f"| tokens | {old.total_tokens:,} | {new.total_tokens:,} | {new.total_tokens - old.total_tokens:+,} |")
    changed = {k: (old.settings.get(k), new.settings.get(k)) for k in {*old.settings, *new.settings}
               if old.settings.get(k) != new.settings.get(k)}
    if changed:
        lines += ["", "Settings that changed:"] + [f"- {k}: `{a}` → `{b}`" for k, (a, b) in sorted(changed.items())]
    return "\n".join(lines) + "\n"


def _fmt(value: float | None) -> str:
    if value is None:
        return "–"
    return f"{value:.3f}" if isinstance(value, float) and not value.is_integer() else f"{value:g}"

