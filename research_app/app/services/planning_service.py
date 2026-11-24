"""
Planning: turn a user's topic into a search plan (field + several search queries).

Borrowed idea (GPT Researcher / Open Deep Research): one vague topic is searched from
several angles. "biomechanics of triceps" -> "triceps brachii EMG activation",
"elbow extension torque", "triceps moment arm" ...
"""

from app.domain.schemas import ResearchPlan
from app.integrations.llm.gateway import LlmGateway

PLAN_PROMPT = """You plan academic literature searches.
Given a research topic, decide its academic field and write search queries for scholarly
databases (OpenAlex, Europe PMC/PubMed, arXiv).

Rules for queries:
- 4 to 5 queries, each 2-5 words, no quotes or boolean operators. Databases match words literally:
  a short query of the core concepts finds far more of the relevant studies than a long sentence.
- MOST queries pair the main subject (the intervention, exposure or structure) with the population or
  context, using the different words studies use for it, e.g. "creatine older men", "creatine elderly
  women", "creatine resistance training older"; or "triceps brachii EMG", "elbow extension torque".
- at most ONE query on mechanisms or measurement methods, and ONE aimed at review papers
  (add the word "review")
- use the technical vocabulary researchers use (e.g. "triceps brachii" not "back of arm")"""


class PlanningService:
    def __init__(self, gateway: LlmGateway):
        self.gateway = gateway

    async def plan(self, topic: str) -> ResearchPlan:
        messages = [{"role": "system", "content": PLAN_PROMPT},
                    {"role": "user", "content": f"Research topic: {topic}"}]
        plan, _ = await self.gateway.complete_structured(messages, ResearchPlan, purpose="plan",
                                                         max_tokens=2000)
        plan.queries = _clean_queries([topic] + plan.queries)   # always also search the topic itself
        return plan


def _clean_queries(queries: list[str], limit: int = 6) -> list[str]:
    """Trim, drop empty and duplicate queries (case-insensitive), keep at most `limit`."""
    seen, cleaned = set(), []
    for q in queries:
        q = " ".join(q.replace('"', " ").split())       # remove quotes and extra spaces
        if q and q.lower() not in seen:
            seen.add(q.lower())
            cleaned.append(q)
    return cleaned[:limit]
