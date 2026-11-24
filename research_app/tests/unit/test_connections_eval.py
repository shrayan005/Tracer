"""Connections eval (step 6.6) with a scripted connection step: no tokens, no database."""

import pytest
import yaml

from app.domain.schemas import Connection, ConnectionMember
from app.evals import connections_eval as ce
from app.evals.runner import Services


def paper(key, n, *claims):
    return {"key": key, "label": f"(Author{key}, 2020)", "study_type": "lab_experiment", "n": 10, "sex": "mixed",
            "duration": "single session",
            "claims": [{"id": f"P{n}.C{m}", "factor": f, "outcome": o, "direction": d, "population": "adults",
                        "magnitude": None} for m, (f, o, d) in enumerate(claims, start=1)]}


PAPERS = [paper(11, 1, ("training (before vs after)", "strength", "increase"), ("grip", "EMG", "increase")),
          paper(22, 2, ("training (before vs after)", "1-RM", "increase")),
          paper(33, 3, ("training (before vs after)", "peak torque", "no difference"), ("grip", "EMG", "decrease"))]
GOLD = {"case": "demo", "brief": "b", "reviewed": True, "papers": PAPERS, "groups": [
    {"question": "Does training raise strength?", "members": [
        {"claim": "P1.C1", "stance": "supports"}, {"claim": "P2.C1", "stance": "supports"},
        {"claim": "P3.C1", "stance": "no effect"}]},
    {"question": "Does grip change EMG?", "members": [
        {"claim": "P1.C2", "stance": "supports"}, {"claim": "P3.C2", "stance": "opposes"}]}]}


def connection(*members):           # (paper key, claim index, stance)
    return Connection(question="q", relation="agree", finding="f",
                      members=[ConnectionMember(paper=p, claim=c, stance=s) for p, c, s in members])


class ScriptedConnector:
    def __init__(self, groups):
        self.groups, self.inputs = groups, None

    async def connect(self, brief, inputs):
        self.inputs = inputs
        return self.groups, []


def test_frozen_claims_become_connection_input():
    key, label, study_type, analysed = ce.as_input(PAPERS[0])
    assert (key, label) == (11, "(Author11, 2020)")
    assert [c.outcome for c in analysed.analysis.claims] == ["strength", "EMG"]


def test_pairs_only_link_different_papers():
    assert ce.pairs([{"P1.C1": "supports", "P1.C2": "supports", "P2.C1": "supports"}]) == {
        frozenset({"P1.C1", "P2.C1"}), frozenset({"P1.C2", "P2.C1"})}


def test_gold_loading_checks_ids_stances_and_papers(tmp_path):
    path = tmp_path / "connections.yaml"
    path.write_text(yaml.safe_dump({"cases": [GOLD]}), encoding="utf-8")
    assert set(ce.load_connections_gold(path)) == {"demo"}
    bad = {**GOLD, "groups": [{"question": "q", "members": [{"claim": "P9.C1", "stance": "supports"},
                                                             {"claim": "P1.C1", "stance": "supports"}]}]}
    path.write_text(yaml.safe_dump({"cases": [bad]}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown claim id 'P9.C1'"):
        ce.load_connections_gold(path)
    one_paper = {**GOLD, "groups": [{"question": "q", "members": [{"claim": "P1.C1", "stance": "supports"},
                                                                   {"claim": "P1.C2", "stance": "supports"}]}]}
    path.write_text(yaml.safe_dump({"cases": [one_paper]}), encoding="utf-8")
    with pytest.raises(ValueError, match="2\\+ papers"):
        ce.load_connections_gold(path)


async def test_suite_scores_links_groups_relations_and_stances(tmp_path, monkeypatch):
    path = tmp_path / "connections.yaml"
    path.write_text(yaml.safe_dump({"cases": [GOLD]}), encoding="utf-8")
    real_loader = ce.load_connections_gold
    monkeypatch.setattr(ce, "load_connections_gold", lambda: real_loader(path))
    # AI: finds the strength group but misses P3.C1 and calls it agree; links the grip claims with a wrong stance
    ai = [connection((11, 0, "supports"), (22, 0, "supports")),
          connection((11, 1, "supports"), (33, 1, "supports"))]
    from app.evals.cases import EvalCase
    case = EvalCase.model_construct(name="demo")
    metrics, details = await ce.connections([case], Services(analysis=None, connection=ScriptedConnector(ai), runs=1))
    # gold pairs: P1C1-P2C1, P1C1-P3C1, P2C1-P3C1, P1C2-P3C2 (4); AI pairs: P1C1-P2C1, P1C2-P3C2 (2, both right)
    assert metrics["pair_precision"] == 1.0 and metrics["pair_recall"] == 0.5
    assert metrics["groups_found"] == 1.0                      # strength: 2/3 overlap; grip: exact
    assert metrics["relation_accuracy"] == 0.0                 # gold: mixed + contradict; AI: agree + agree
    assert metrics["stance_accuracy"] == pytest.approx(3 / 4)  # P3.C2 should be "opposes"
    assert any(d.startswith("MISSED LINK") and "P3.C1" in d for d in details)
