"""
Command line (presentation layer, like the HTTP controllers).

    python -m app.cli health [--deep]
    python -m app.cli ask "What is a token?"            # one real LLM call through the gateway
    python -m app.cli usage                              # recent LLM calls + total cost
    python -m app.cli search "biomechanics of triceps"   # discover papers for a topic
    python -m app.cli research "biomechanics of triceps" # full pipeline -> reports/<id>-<topic>.md
    python -m app.cli report 12                          # rebuild topic 12's report (no LLM calls)
    python -m app.cli fetch 12 [--retry]                 # download + extract free full texts of topic 12's papers
    python -m app.cli analyze 12 [--refresh]             # claims, quality, connections, gaps (stored)
    python -m app.cli paper 12 3                         # detailed page for the paper ranked 3 in topic 12
    python -m app.cli index 12 [--refresh]               # prepare topic 12's papers for questions
    python -m app.cli jobs submit research "topic"       # queue a background job (worker: arq app.worker.WorkerSettings)
    python -m app.cli jobs status 42 | jobs list | jobs retry 42
    python -m app.cli users set-password local@localhost  # let the CLI's user log in to the web app
    python -m app.cli users add you@example.com --name You

The app is async, so main() starts ONE event loop with asyncio.run() and everything
inside run() can await.
"""

import argparse
import asyncio
import getpass
import json
import sys
from pathlib import Path

from app.common.logging import setup_logging
from app.common.progress import printer
from app.integrations.llm.run_context import user_scope
from app.config import get_settings
from app.dependencies import (close_resources, get_auth_service, get_current_user_id, get_discovery_service,
                              get_extraction_service, get_fulltext_fetcher, get_health_service, get_job_service,
                              get_index_service, get_llm_gateway, get_research_service)
from app.domain.errors import (ConflictError, ForbiddenError, InvalidInputError, NotFoundError,
                               QuotaExceededError)
from app.domain.schemas import (DiscoveryResult, EvidenceGaps, ExtractionResult, FetchResult, JobInfo,
                                PaperDetail, ResearchReport)
from app.presentation.paper_markdown import render_paper
from app.presentation.report_markdown import render, slugify
from app.repositories.llm_call_repository import LlmCallRepository
from app.repositories.session import session_scope


def main() -> None:
    setup_logging(get_settings().log_level)
    parser = argparse.ArgumentParser(prog="app.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    h = sub.add_parser("health", help="check database, redis and LLM providers")
    h.add_argument("--deep", action="store_true", help="also make a tiny real LLM call")
    a = sub.add_parser("ask", help="send one question through the LLM gateway")
    a.add_argument("question", nargs="+")
    sub.add_parser("usage", help="show recent LLM calls and total cost")
    s = sub.add_parser("search", help="discover papers for a topic")
    s.add_argument("topic", nargs="+")
    s.add_argument("--show", type=int, default=15, help="how many candidates to print")
    r = sub.add_parser("research", help="full pipeline: find, screen, select, summarise -> Markdown report")
    r.add_argument("topic", nargs="+")
    r.add_argument("--out", default=None, help="folder for the report (default: REPORTS_DIR)")
    r.add_argument("--papers", type=int, choices=[5, 10, 15, 20], default=10,
                   help="how many papers to pick, read and prepare for questions (default 10)")
    rp = sub.add_parser("report", help="rebuild a topic's report from the database (no LLM calls)")
    rp.add_argument("topic_id", type=int)
    rp.add_argument("--out", default=None, help="folder for the report (default: REPORTS_DIR)")
    f = sub.add_parser("fetch", help="download the free full texts of a topic's selected papers")
    f.add_argument("topic_id", type=int)
    f.add_argument("--retry", action="store_true", help="also retry papers that failed recently")
    an = sub.add_parser("analyze", help="claims, quality, connections and gaps for a topic (stored)")
    an.add_argument("topic_id", type=int)
    an.add_argument("--refresh", action="store_true", help="re-analyse even if a stored analysis is still valid")
    pp = sub.add_parser("paper", help="detailed page for one paper of a topic (no LLM calls)")
    pp.add_argument("topic_id", type=int)
    pp.add_argument("rank", type=int, help="the paper's rank in the topic's report (1 = best)")
    pp.add_argument("--out", default=None, help="folder for the page (default: REPORTS_DIR)")
    ix = sub.add_parser("index", help="prepare a topic's papers for questions (passages + embeddings)")
    ix.add_argument("topic_id", type=int)
    ix.add_argument("--refresh", action="store_true", help="re-index even if already indexed")
    j = sub.add_parser("jobs", help="background jobs (run the worker: arq app.worker.WorkerSettings)")
    jsub = j.add_subparsers(dest="jobs_command", required=True)
    js = jsub.add_parser("submit", help="queue a job")
    js.add_argument("kind", choices=["research", "analyze", "index"])
    js.add_argument("what", nargs="+", help="research: the topic · analyze: the topic id")
    js.add_argument("--refresh", action="store_true", help="analyze: re-analyse even if still valid")
    jst = jsub.add_parser("status", help="one job: status, progress, cost")
    jst.add_argument("job_id", type=int)
    jsub.add_parser("list", help="your recent jobs")
    jsub.add_parser("usage", help="LLM tokens used today against the daily budget")
    jr = jsub.add_parser("retry", help="start a failed job again (as a new job)")
    jr.add_argument("job_id", type=int)
    u = sub.add_parser("users", help="accounts for the web app (login)")
    usub = u.add_subparsers(dest="users_command", required=True)
    ua = usub.add_parser("add", help="create an account (asks for the password)")
    ua.add_argument("email")
    ua.add_argument("--name", default=None, help="display name")
    up = usub.add_parser("set-password", help="set or reset a password (e.g. for local@localhost, who owns "
                                              "the topics made with the CLI)")
    up.add_argument("email")
    args = parser.parse_args()

    asyncio.run(run(args))     # start the event loop, run the command, then stop the loop


async def run(args: argparse.Namespace) -> None:
    try:
        with user_scope(None if args.command in ("health", "users") else await get_current_user_id()):
            await dispatch(args)        # LLM calls made here count toward the local user's budget (7.4)
    finally:
        await close_resources()   # close DB/Redis/HTTP connections before the event loop ends


async def dispatch(args: argparse.Namespace) -> None:
    if args.command == "health":
        report = await get_health_service().check(deep=args.deep)
        print(json.dumps(report.model_dump(), indent=2))
    elif args.command == "ask":
        result = await get_llm_gateway().complete(
            [{"role": "user", "content": " ".join(args.question)}], purpose="cli_ask")
        print(result.text)
        print(f"\n[{result.provider} · {result.model} · in {result.input_tokens} / out "
              f"{result.output_tokens} / thinking {result.thinking_tokens} · {result.latency_s}s · "
              f"cost {result.cost_usd} · attempts {result.attempts}]")
    elif args.command == "usage":
        async with session_scope() as session:
            repo = LlmCallRepository(session)
            for c in await repo.recent(15):
                print(f"{c.created_at:%Y-%m-%d %H:%M:%S} {c.purpose:<10} {c.provider:<10} {c.status:<5} "
                      f"in {c.input_tokens:>5} out {c.output_tokens:>5} think {c.thinking_tokens:>5} "
                      f"{c.latency_ms:>6}ms {c.error or ''}"[:160])
            print(f"Total cost: ${await repo.total_cost():.6f}")
    elif args.command == "search":
        result = await get_discovery_service().discover(" ".join(args.topic), await get_current_user_id())
        print_discovery(result, args.show)
    elif args.command == "research":
        report = await get_research_service().research(" ".join(args.topic), await get_current_user_id(),
                                                       progress=printer(), papers=args.papers)
        save_report(report, args.out)
    elif args.command == "report":
        try:
            report = await get_research_service().load(args.topic_id, await get_current_user_id())
        except NotFoundError as e:
            sys.exit(f"error: {e}")
        save_report(report, args.out)
    elif args.command == "fetch":
        try:
            report = await get_research_service().load(args.topic_id, await get_current_user_id())
        except NotFoundError as e:
            sys.exit(f"error: {e}")
        results = await get_fulltext_fetcher().fetch_many({p.paper_id: p.paper for p in report.papers},
                                                          retry_failed=args.retry)
        extractor = get_extraction_service()
        extracted = {pid: await extractor.extract(r.document_id)          # one at a time: CPU work
                     for pid, r in results.items() if r.document_id is not None}
        print_fetch(report, results, extracted)
    elif args.command == "analyze":
        service, user_id = get_research_service(), await get_current_user_id()
        try:
            report = await service.load(args.topic_id, user_id)
            if args.refresh or report.gaps is None:          # not analysed yet (or asked to redo it)
                report = await service.analyze_topic(args.topic_id, user_id, refresh=args.refresh,
                                                     progress=printer())
        except NotFoundError as e:
            sys.exit(f"error: {e}")
        print_analysis(report)
        print_connections(report)
        if report.gaps is not None:
            print_gaps(report.gaps)
    elif args.command == "paper":
        try:
            detail = await get_research_service().paper_detail(args.topic_id, args.rank,
                                                               await get_current_user_id())
        except NotFoundError as e:
            sys.exit(f"error: {e}")
        save_paper(detail, args.out)
    elif args.command == "index":
        try:
            report = await get_research_service().load(args.topic_id, await get_current_user_id())
        except NotFoundError as e:
            sys.exit(f"error: {e}")
        result = await get_index_service().index([p.paper_id for p in report.papers], refresh=args.refresh,
                                                 progress=printer())
        print(f"topic #{report.topic_id}: {result.indexed} papers indexed ({result.passages} passages), "
              f"{result.reused} already indexed, {result.empty} without any text")
        for failure in result.failed:
            print("  failed:", failure)
    elif args.command == "jobs":
        await run_jobs(args, await get_current_user_id())
    elif args.command == "users":
        await run_users(args)


async def run_jobs(args: argparse.Namespace, user_id: int) -> None:
    jobs = get_job_service()
    try:
        if args.jobs_command == "submit":
            if args.kind == "research":
                params = {"topic": " ".join(args.what)}
            else:
                if len(args.what) != 1 or not args.what[0].isdigit():
                    sys.exit(f"error: {args.kind} needs one topic id, e.g. jobs submit {args.kind} 119")
                params = {"topic_id": int(args.what[0])} | ({} if args.kind == "index" else {"refresh": args.refresh})
            job = await jobs.submit(args.kind, params, user_id)
            print(f"job {job.id} queued ({job.kind} {params}). Follow it: python -m app.cli jobs status {job.id}")
        elif args.jobs_command == "status":
            print_job(await jobs.get(args.job_id, user_id))
        elif args.jobs_command == "list":
            for job in await jobs.list(user_id):
                what = job.params.get("topic") or f"topic #{job.params.get('topic_id')}"
                print(f"{job.id:>5} {job.status:<8} {job.kind:<8} {job.created_at:%Y-%m-%d %H:%M} {what[:60]}"
                      f"{f' -> topic #{job.topic_id}' if job.topic_id and job.kind == 'research' else ''}")
        elif args.jobs_command == "usage":
            u = await jobs.usage(user_id)
            limit = f"{u.daily_budget:,} · {u.remaining:,} left" if u.daily_budget else "no limit"
            print(f"today: {u.tokens_today:,} tokens · ${u.cost_today_usd:.4f} · budget {limit} · "
                  f"resets {u.resets_at:%Y-%m-%d %H:%M} UTC")
        elif args.jobs_command == "retry":
            job = await jobs.retry(args.job_id, user_id)
            print(f"job {job.id} queued (a retry of job {args.job_id})")
    except (NotFoundError, ConflictError, QuotaExceededError, ValueError) as e:
        sys.exit(f"error: {e}")


async def run_users(args: argparse.Namespace) -> None:
    auth = get_auth_service()
    if args.users_command == "set-password":
        await get_current_user_id()                       # makes sure the default local user exists
    password = getpass.getpass("password (10+ characters): ")
    if getpass.getpass("again: ") != password:
        sys.exit("error: the passwords differ")
    try:
        if args.users_command == "add":
            user = await auth.register(args.email, password, args.name)
            print(f"created user {user.id} <{user.email}>")
        else:
            user = await auth.set_password(args.email, password)
            print(f"password set for user {user.id} <{user.email}>")
    except (ConflictError, ForbiddenError, InvalidInputError) as e:
        sys.exit(f"error: {e}")


def print_job(job: JobInfo) -> None:
    took = f" · took {(job.finished_at - job.started_at).total_seconds():.0f}s" if job.finished_at and job.started_at else ""
    print(f"job {job.id} · {job.kind} {job.params} · {job.status.upper()}{took}"
          f"{f' · retry of job {job.retry_of}' if job.retry_of else ''}")
    for p in job.progress:
        print(f"  {p.at[11:19]} · {p.message}")
    if job.status == "done" and job.topic_id:
        print(f"result: topic #{job.topic_id} · python -m app.cli report {job.topic_id}")
    if job.error:
        print(f"error: {job.error}\n(start it again: python -m app.cli jobs retry {job.id})")
    print(f"LLM use: {job.tokens:,} tokens · ${job.cost_usd:.4f}")


def save_report(report: ResearchReport, out_dir: str | None) -> Path:
    """Write the Markdown report to <out_dir>/<topic id>-<topic>.md and print a short overview."""
    folder = Path(out_dir or get_settings().reports_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{report.topic_id}-{slugify(report.topic)}.md"
    path.write_text(render(report), encoding="utf-8")

    print(f"\nTopic #{report.topic_id} · {report.plan.field} · {len(report.papers)} papers selected")
    for warning in report.warnings:
        print("  ⚠", warning)
    for p in report.papers:
        print(f"  {p.rank:>2}. {p.score:.2f} rel {p.relevance:>2} {p.trust.label:<9} {p.in_text:<28} "
              f"{p.paper.title[:60]}")
    print(f"\nReport: {path}")
    return path


def save_paper(detail: PaperDetail, out_dir: str | None) -> Path:
    """Write one paper's detail page to <out_dir>/<topic>-paper-<rank>-<title>.md and print a summary."""
    folder = Path(out_dir or get_settings().reports_dir)
    folder.mkdir(parents=True, exist_ok=True)
    p = detail.paper
    path = folder / f"{detail.topic_id}-paper-{p.rank}-{slugify(p.paper.title, 40)}.md"
    path.write_text(render_paper(detail), encoding="utf-8")
    base = p.base_trust or p.trust
    claims = len(p.analysis.analysis.claims) if p.analysis else 0
    print(f"\n{p.rank}. {p.in_text} {p.paper.title[:80]}")
    print(f"   trust {base.label} → {p.trust.label} · {claims} claims · {len(detail.connections)} connections · "
          f"{'full text' if detail.document and detail.document.status == 'extracted' else 'abstract only'}")
    print(f"\nPage: {path}")
    return path


def print_fetch(report: ResearchReport, results: dict[int, FetchResult],
                extracted: dict[int, ExtractionResult]) -> None:
    print(f"\nTopic #{report.topic_id}: full texts")
    for p in report.papers:
        r = results[p.paper_id]
        if r.document_id is None:
            print(f"  {p.rank:>2}. ✗ {p.paper.title[:55]:<55} {'; '.join(r.failures)[:100]}")
            continue
        note = "already stored" if r.reused else f"via {r.via}"
        print(f"  {p.rank:>2}. ✓ {p.paper.title[:55]:<55} {r.kind} {(r.size_bytes or 0) // 1024:>5} KB  {note}")
        if r.failures:
            print(f"       (tried first: {'; '.join(r.failures)[:110]})")
        e = extracted[p.paper_id]
        if e.status == "extracted":
            print(f"       extracted{' (earlier)' if e.reused else ''}: {e.sections} sections, "
                  f"{e.characters // 1000}k chars: {', '.join(e.kinds)}")
        else:
            print(f"       extraction failed: {e.error}")
    stored = sum(r.document_id is not None for r in results.values())
    readable = sum(e.status == "extracted" for e in extracted.values())
    print(f"\n{stored} of {len(results)} papers have a stored full text, {readable} extracted to sections")


ARROWS = {"increase": "↑", "decrease": "↓", "no difference": "=", "association": "~", "other": "·"}
FACT_NAMES = {"control_group": "control", "randomized": "randomised", "blinded": "blinded",
              "statistics_reported": "statistics", "effect_sizes": "effect sizes",
              "conflicts_declared": "conflicts stated", "data_available": "data available"}


def print_analysis(report: ResearchReport) -> None:
    print(f"\nTopic #{report.topic_id}: claims and appraisal facts")
    for p in report.papers:
        result = p.analysis
        print(f"\n{p.rank:>2}. {p.in_text} {p.paper.title[:70]}")
        if result is None:
            print("    (not analysed: no text)")
            continue
        for c in result.analysis.claims:
            print(f"    {ARROWS[c.direction]} {c.factor} → {c.outcome}"
                  f"{f' ({c.magnitude})' if c.magnitude else ''} | {c.population}")
        a = result.analysis.appraisal
        facts = " · ".join(f"{name} {'✓' if getattr(a, key).value == 'yes' else '✗' if getattr(a, key).value == 'no' else '?'}"
                           for key, name in FACT_NAMES.items() if getattr(a, key).value != "not applicable")
        print(f"    n={a.sample_size if a.sample_size is not None else '?'} · {a.sex} · {a.duration} · {facts}"
              f"   [{result.based_on}]")
        base = p.base_trust or p.trust
        print(f"    trust: {base.label} ({base.level}) → {p.trust.label} ({p.trust.level}) · "
              f"{' · '.join(p.trust.reasons[-2:])}")
        for flag in result.flags:
            print(f"    ⚠ {flag}")
    print("\n✓ = yes (quoted)  ✗ = no (quoted)  ? = not reported")


RELATION_ICONS = {"agree": "✓ agree", "contradict": "✗ contradict", "mixed": "≈ mixed"}


def print_connections(report: ResearchReport) -> None:
    names = {p.paper_id: p.in_text for p in report.papers}
    print("\n── How the papers connect ──")
    if not report.connections:
        print("(no comparable findings across papers)")
    for c in report.connections:
        print(f"\n{RELATION_ICONS[c.relation]}: {c.question}")
        for m in c.members:
            claim = m.claim_snapshot                      # the claim exactly as it was connected
            if claim is None:
                continue
            print(f"   {ARROWS[claim.direction]} {m.stance:<9} {names.get(m.paper, '?'):<30} "
                  f"{claim.factor[:40]} → {claim.outcome[:40]}")
        print(f"   finding: {c.finding}")
        if c.why_differ:
            print(f"   why they differ: {c.why_differ}")
    for flag in report.synthesis_flags:
        print("⚠", flag)


def print_gaps(gaps: EvidenceGaps) -> None:
    print("\n── What the evidence is missing ──")
    for gap in gaps.gaps or ["(no gaps found by the rules)"]:
        print(f" - {gap}")
    for note in gaps.notes:
        print(f"   note: {note}")
    print(f"\nLimitations the authors admit ({gaps.papers_with_limitations} of {gaps.n_papers} papers state any):")
    for shared in gaps.shared_limitations:
        print(f" - {shared.theme} ({len(shared.papers)} papers): {', '.join(shared.papers)}")
    if not gaps.shared_limitations:
        print(" - none shared by 2 or more papers")
    if gaps.write_up:
        print(f"\n{gaps.write_up.summary}")
        print("Studies that would fill these gaps:")
        for study in gaps.write_up.next_studies:
            print(f" - {study}")
    for flag in gaps.flags:
        print("⚠", flag)


def print_discovery(result: DiscoveryResult, show: int) -> None:
    """Pretty-print a discovery result (normal function: it only prints, no waiting)."""
    plan = result.plan
    print(f"\nTopic #{result.topic_id} · field: {plan.field}")
    print(f"Brief: {plan.research_brief}")
    print("Queries:", " | ".join(plan.queries))
    print(f"Sources: {', '.join(result.sources_used)} · raw results {result.raw_results} "
          f"→ {result.unique_papers} unique papers · per source {result.per_source}")
    for error in result.errors:
        print("  ⚠", error)
    print()
    for i, c in enumerate(result.candidates[:show], start=1):
        p = c.paper
        sources = sorted({f.source for f in c.found_by})
        print(f"{i:>2}. [{p.year or '????'}] {p.title[:90]}")
        print(f"     {p.venue or '-'} · {p.publication_type or '-'} · cited {p.citation_count or 0} · "
              f"found {len(c.found_by)}x via {', '.join(sources)} · abstract {'yes' if p.abstract else 'no'}"
              f"{' · RETRACTED' if p.is_retracted else ''}")


if __name__ == "__main__":
    main()
