"""Search agent decision step (no network): round summary, code checks on the LLM's queries."""

import json

from app.domain.schemas import Candidate, FoundBy, PaperRecord
from app.integrations.llm.gateway import LlmGateway
from app.services.search_agent import SearchAgent, clean_new, round_summary
from tests.fakes import FakeProvider


async def no_sleep(_seconds):
    pass


def cand(title, *queries):
    return Candidate(paper=PaperRecord(source="openalex", title=title),
                     found_by=[FoundBy(source="openalex", query=q) for q in queries])


def test_round_summary_counts_found_and_only():
    cands = [cand("a", "q1", "q2"), cand("b", "q1"), cand("c", "q2")]
    assert round_summary(cands, ["q1", "q2", "q3"]) == {"q1": (2, 1), "q2": (2, 1), "q3": (0, 0)}


def test_clean_new_drops_repeats_long_and_extra():
    tried = ["creatine older adults"]
    suggested = ["Creatine  older adults",                       # repeat (case/spacing)
                 "creatine supplementation combined with resistance training in older men",   # too long
                 "creatine older men", "creatine older women", "creatine older men",          # duplicate
                 "creatine whey protein older", "creatine sarcopenia"]                        # 4th: cut
    assert clean_new(suggested, tried) == ["creatine older men", "creatine older women",
                                           "creatine whey protein older"]


async def test_review_with_no_usable_queries_means_done():
    reply = json.dumps({"gaps": "none", "done": False, "new_queries": ["creatine older adults"]})
    agent = SearchAgent(LlmGateway([FakeProvider("fake", [reply])], sleep=no_sleep))
    review = await agent.review("creatine in older adults", {"creatine older adults": (10, 3)}, ["t"], limit=15)
    assert review.new_queries == [] and review.done is True
