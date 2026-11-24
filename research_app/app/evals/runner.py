"""
Eval runner: run one suite on frozen cases inside a run scope, then build a scorecard.

A suite is an async function (cases, services) -> (metrics, details). Every LLM call it makes is
tagged with the run id (run_context), so the cost comes straight from llm_calls.
"""

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import func, select

from app.config import get_settings
from app.integrations.llm.gateway import LlmGateway
from app.repositories.models import LlmCall
from app.repositories.session import session_scope
from app.services.analysis_service import ANALYSIS_VERSION, AnalysisService
from app.services.ask_service import AskService
from app.services.connection_service import ConnectionService
from app.services.discovery_service import DiscoveryService
from app.services.screening_service import ScreeningService

from app.evals.cases import EvalCase
from app.evals.scorecard import Scorecard, StageCost
from app.integrations.llm.run_context import run_scope


@dataclass
class Services:
    """What suites may use (built by the CLI from dependencies.py; fakes in tests)."""
    analysis: AnalysisService
    connection: ConnectionService | None = None
    runs: int = 3                                  # repetitions, for suites that repeat (stability)
    judge: LlmGateway | None = None                # independent judge gateway (grounding suites)
    screener: ScreeningService | None = None       # screening + search-recall suites (step 6.5)
    discovery: DiscoveryService | None = None      # search-recall suite: live search, nothing saved
    replan: bool = False                           # search-recall: ask the planner again (test a plan prompt)
    asker: AskService | None = None                # ask suite (7.7): retrieval + checked answers


SuiteResult = tuple[dict[str, float], list[str]]
Suite = Callable[[list[EvalCase], Services], Awaitable[SuiteResult]]
SUITES: dict[str, Suite] = {}


def suite(name: str):
    """Decorator: register an eval suite under a name."""
    def register(fn: Suite) -> Suite:
        SUITES[name] = fn
        return fn
    return register


def settings_snapshot() -> dict[str, str]:
    s = get_settings()
    return {"analysis_version": ANALYSIS_VERSION, "judge": s.judge_provider,
            "provider_order": ",".join(s.llm_provider_order),
            "routes": ",".join(f"{k}->{'/'.join(v)}" for k, v in sorted(s.llm_routes.items())),
            "groq_model": s.groq_model, "gemini_model": s.gemini_model,
            "results_per_query": str(s.results_per_query), "prefilter_keep": str(s.prefilter_keep)}


async def cost_of(run_id: int) -> dict[str, StageCost]:
    """Cost per LLM purpose AND provider ("analyze · gemini"): a quiet fallback to another model can
    change results, so the scorecard shows which model actually did the work."""
    async with session_scope() as session:
        rows = await session.execute(
            select(LlmCall.purpose, LlmCall.provider, func.count(),
                   func.sum(LlmCall.input_tokens + LlmCall.output_tokens + LlmCall.thinking_tokens),
                   func.sum(LlmCall.latency_ms))
            .where(LlmCall.run_id == run_id, LlmCall.status == "ok")
            .group_by(LlmCall.purpose, LlmCall.provider).order_by(LlmCall.purpose, LlmCall.provider))
        return {cost_key(purpose, provider): StageCost(calls=calls, tokens=tokens or 0, seconds=(ms or 0) / 1000)
                for purpose, provider, calls, tokens, ms in rows}


def cost_key(purpose: str, provider: str) -> str:
    return f"{purpose} · {provider}"


async def run_suite(name: str, cases: list[EvalCase], services: Services) -> Scorecard:
    if name not in SUITES:
        raise KeyError(f"unknown suite '{name}' (known: {', '.join(sorted(SUITES))})")
    started, clock = datetime.now(timezone.utc), time.monotonic()
    run_id = int(started.timestamp())                       # unique enough for runs started by hand
    with run_scope(run_id):
        metrics, details = await SUITES[name](cases, services)
    return Scorecard(suite=name, run_id=run_id, started_at=started, seconds=round(time.monotonic() - clock, 1),
                     cases=[c.name for c in cases], settings=settings_snapshot(), metrics=metrics,
                     details=details, cost=await cost_of(run_id))


# ---------- a tiny suite to prove the plumbing (1 LLM call per case) ----------

@suite("smoke")
async def smoke(cases: list[EvalCase], services: Services) -> SuiteResult:
    """Analyse the first paper of each case once: checks cases load, the run id tags calls, cost is counted."""
    analysed = claims = flags = 0
    details = []
    for case in cases:
        first = case.papers[0]
        text = first.text()
        results, warnings = await services.analysis.analyze({first.key: first.paper},
                                                             {first.key: text} if text else None)
        details += warnings
        if first.key in results:
            analysed += 1
            claims += len(results[first.key].analysis.claims)
            flags += len(results[first.key].flags)
            details.append(f"{case.name}: {first.label} → {len(results[first.key].analysis.claims)} claims, "
                           f"{len(results[first.key].flags)} flags ({results[first.key].based_on})")
    return {"papers_analysed": analysed, "claims": claims, "flags": flags}, details
