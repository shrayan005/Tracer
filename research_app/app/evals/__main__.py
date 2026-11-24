"""
Eval command line (milestone 6).

    python -m app.evals freeze 119 triceps        save topic 119's papers (+ full text) as a fixed test case
    python -m app.evals cases                     list frozen cases
    python -m app.evals run smoke                 run a suite on all cases -> scorecard in evals/results/
    python -m app.evals run smoke --cases triceps
    python -m app.evals run stability [--runs 3]  same input N times: what changes? (no labels needed)
    python -m app.evals run grounding             independent judge: do quotes support the claims? (+ planted errors)
    python -m app.evals label <items.yaml>        make evals/gold/judge_check.yaml for you to label
    python -m app.evals run judge-check           how often does the judge agree with your labels?
    python -m app.evals prefill appraisal         make evals/gold/appraisal.yaml (AI facts + hints) for you to check
    python -m app.evals run appraisal             AI's quality facts vs your gold: accuracy, false/missed ✓, trust
    python -m app.evals freeze 119 triceps --force   re-freeze (cases now also keep the screened candidates)
    python -m app.evals prefill screening         make evals/gold/screening.yaml (AI's buckets) for you to check
    python -m app.evals run screening             current screening vs your buckets: missed relevant papers
    python -m app.evals prefill recall creatine   make evals/gold/recall_creatine.yaml from the reviews' references
    python -m app.evals run search-recall         LIVE search: where are the experts' studies lost? per source
    python -m app.evals prefill connections       make evals/gold/connections.yaml (frozen claims + AI's groups)
    python -m app.evals run connections --runs 3  connection step vs your groups: links, groups, stances
    python -m app.evals run ask                   questions over papers: retrieval, answers, declines, quotes (7.7)
    python -m app.evals compare A.json B.json     what changed between two runs

Evals cost tokens and take minutes, so they are run on purpose (before and after a change),
not as part of pytest.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from app.common.logging import setup_logging
from app.config import get_settings
from app.dependencies import (close_resources, get_analysis_service, get_ask_service, get_connection_service, get_current_user_id,
                              get_discovery_service, get_judge_gateway, get_research_service, get_screening_service,
                              get_source_clients)
from app.domain.errors import NotFoundError
from app.evals import (appraisal, ask_eval, connections_eval, grounding, recall, screening_eval,  # noqa: F401
                       stability)
from app.evals.connections_eval import prefill_connections
from app.evals.appraisal import prefill as prefill_appraisal
from app.evals.recall import prefill_recall
from app.evals.screening_eval import prefill_screening
from app.evals.cases import freeze, list_cases, load_case
from app.evals.labels import make_label_file
from app.evals.runner import SUITES, Services, run_suite
from app.evals.scorecard import compare, load_scorecard, render_scorecard, save_scorecard


def main() -> None:
    setup_logging(get_settings().log_level)
    parser = argparse.ArgumentParser(prog="app.evals")
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze", help="save a researched topic's papers as a fixed eval case")
    f.add_argument("topic_id", type=int)
    f.add_argument("name", help="short case name, e.g. triceps")
    f.add_argument("--force", action="store_true", help="replace an existing case (gold files refer to cases)")
    sub.add_parser("cases", help="list frozen cases")
    r = sub.add_parser("run", help="run an eval suite and save a scorecard")
    r.add_argument("suite", help=f"one of: {', '.join(sorted(SUITES))}")
    r.add_argument("--cases", nargs="+", default=None, help="case names (default: all)")
    r.add_argument("--runs", type=int, default=3, help="repetitions for suites that repeat (stability)")
    r.add_argument("--replan", action="store_true", help="search-recall: plan again instead of the stored plan")
    c = sub.add_parser("compare", help="compare two scorecards (JSON files)")
    c.add_argument("old")
    c.add_argument("new")
    lb = sub.add_parser("label", help="make the file of judged items for YOU to label (judge check)")
    lb.add_argument("items", help="an items file from a grounding run (evals/results/<run>-grounding-items.yaml)")
    lb.add_argument("--sample", type=int, default=30, help="how many items to label")
    lb.add_argument("--force", action="store_true", help="replace an existing label file (loses its labels!)")
    pf = sub.add_parser("prefill", help="make a pre-filled gold file for YOU to check")
    pf.add_argument("what", choices=["appraisal", "screening", "recall", "connections"])
    pf.add_argument("case", nargs="?", help="for recall: the case whose reviews' references become the gold")
    pf.add_argument("--cases", nargs="+", default=None, help="case names (default: all)")
    pf.add_argument("--force", action="store_true", help="replace an existing gold file (loses its labels!)")
    asyncio.run(run(parser.parse_args()))


async def run(args: argparse.Namespace) -> None:
    try:
        if args.command == "freeze":
            try:
                case = await freeze(get_research_service(), args.topic_id, args.name, await get_current_user_id(),
                                    force=args.force)
            except (NotFoundError, FileExistsError) as e:
                sys.exit(f"error: {e}")
            full = sum(bool(p.sections) for p in case.papers)
            print(f"froze '{case.name}': topic #{case.topic_id} \"{case.topic}\" · {len(case.papers)} papers "
                  f"({full} with full text) · {len(case.candidates)} screened candidates")
        elif args.command == "cases":
            for name in list_cases() or ["(none yet: python -m app.evals freeze <topic id> <name>)"]:
                print(name)
        elif args.command == "run":
            if args.suite not in SUITES:
                sys.exit(f"error: unknown suite '{args.suite}' (known: {', '.join(sorted(SUITES))})")
            names = args.cases or list_cases()
            if not names:
                sys.exit("error: no frozen cases (python -m app.evals freeze <topic id> <name>)")
            try:
                cases = [load_case(n) for n in names]
            except FileNotFoundError as e:
                sys.exit(f"error: {e}")
            services = Services(analysis=get_analysis_service(), connection=get_connection_service(),
                                runs=max(1, args.runs), judge=get_judge_gateway(),
                                screener=get_screening_service(), discovery=get_discovery_service(),
                                replan=args.replan, asker=get_ask_service())
            try:
                card = await run_suite(args.suite, cases, services)
            except (FileNotFoundError, ValueError) as e:      # e.g. labels missing or unfinished
                sys.exit(f"error: {e}")
            json_path, md_path = save_scorecard(card)
            print(render_scorecard(card))
            print(f"Saved: {md_path}\n       {json_path}")
        elif args.command == "compare":
            print(compare(load_scorecard(args.old), load_scorecard(args.new)))
        elif args.command == "prefill":
            try:
                if args.what == "recall":
                    if not args.case:
                        sys.exit("error: name the case, e.g. python -m app.evals prefill recall creatine")
                    path, n, marked = await prefill_recall(load_case(args.case), get_source_clients()["openalex"],
                                                           force=args.force)
                    print(f"wrote {n} references ({marked} pre-marked include: true) to {path}\nConfirm the "
                          f"`include:` marks, set `reviewed: true`, then run: python -m app.evals run search-recall")
                    return
                cases = [load_case(n) for n in (args.cases or list_cases())]
                if args.what == "connections":
                    path, n = await prefill_connections(
                        cases, Services(analysis=None, connection=get_connection_service()), force=args.force)
                    print(f"wrote {n} proposed groups to {path}\nCorrect the groups, set `reviewed: true` per case, "
                          f"then run: python -m app.evals run connections --runs 3")
                    return
                if args.what == "screening":
                    path, n = prefill_screening(cases, force=args.force)
                    print(f"wrote {n} screened papers to {path}\nCorrect the `gold:` buckets, set `reviewed: true` "
                          f"per case, then run: python -m app.evals run screening")
                    return
                path, n = await prefill_appraisal(cases, Services(analysis=get_analysis_service()), force=args.force)
            except (FileExistsError, FileNotFoundError, ValueError) as e:
                sys.exit(f"error: {e}")
            print(f"wrote {n} papers to {path}\nFor each paper: correct any wrong `gold:` values, set "
                  f"`reviewed: true`, then run: python -m app.evals run appraisal")
        elif args.command == "label":
            try:
                path, n = make_label_file(Path(args.items), sample=args.sample, force=args.force)
            except FileExistsError as e:
                sys.exit(f"error: {e}")
            print(f"wrote {n} items to {path}\nOpen it, set `human:` for each item, then run: "
                  f"python -m app.evals run judge-check")
    finally:
        await close_resources()


if __name__ == "__main__":
    main()
