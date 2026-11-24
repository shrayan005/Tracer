"""
Questions over papers (step 7.7): does retrieval find the right passages, and are the answers grounded?

Gold (evals/gold/ask.yaml): questions with the case paper(s) that hold the answer, plus questions the papers do
NOT cover. The suite asks each one over the frozen case's papers (their stored passages), with the real
retrieval and the real answer step, and measures:

    retrieval_recall      an expected paper is among the passages found (the answer can't be better than this)
    answered_correctly    answerable questions answered AND citing an expected paper
    declined_correctly    unanswerable questions declined (answered=false), not guessed
    quotes_ok             answer sentences whose quote AND numbers passed the code check
"""

from app.evals.cases import EvalCase
from app.evals.labels import GOLD_DIR, read_yaml
from app.evals.runner import Services, SuiteResult, suite
from app.services.ask_service import focus

ASK = GOLD_DIR / "ask.yaml"


@suite("ask")
async def ask(cases: list[EvalCase], services: Services) -> SuiteResult:
    if services.asker is None:
        raise ValueError("the ask suite needs the ask service")
    gold = (read_yaml(ASK) or {}).get("questions", [])
    by_name = {c.name: c for c in cases}
    found = answerable = right = unanswerable = declined = sentences = sentences_ok = 0
    details = []
    for q in gold:
        case = by_name.get(q["case"])
        if case is None:
            continue
        papers = {p.key: (i, p.label) for i, p in enumerate(case.papers, start=1)}
        hits = await services.asker.retrieval.search(q["question"], focus(q["question"], papers), k=8)
        answer = await services.asker._answer(q["question"], hits, papers, [])
        expect = set(q.get("expect") or [])
        cited = {p.citation.paper_id for p in answer.parts if p.citation}
        sentences += len(answer.parts)
        sentences_ok += sum(not p.flags and p.citation is not None for p in answer.parts)
        tag = f"{q['case']} · {q['question'][:70]}"
        if q.get("answerable", True):
            answerable += 1
            hit = bool(expect & {h.paper_id for h in hits})
            found += hit
            ok = answer.answered and bool(expect & cited)
            right += ok
            if not hit:
                details.append(f"NOT RETRIEVED · {tag}")
            elif not ok:
                details.append(f"WRONG/NO ANSWER · {tag} → answered={answer.answered}, cited "
                               f"{sorted(papers[c][1] for c in cited)}")
        else:
            unanswerable += 1
            declined += not answer.answered
            if answer.answered:
                details.append(f"GUESSED · {tag} → {' '.join(p.text for p in answer.parts)[:120]}")
        details += [f"FLAG · {tag} · {f}" for p in answer.parts for f in p.flags]
    share = lambda a, b: a / b if b else 0.0                       # noqa: E731
    return {"questions": float(answerable + unanswerable), "retrieval_recall": share(found, answerable),
            "answered_correctly": share(right, answerable), "declined_correctly": share(declined, unanswerable),
            "quotes_ok": share(sentences_ok, sentences), "sentences_per_answer": share(sentences, answerable + unanswerable)}, details
