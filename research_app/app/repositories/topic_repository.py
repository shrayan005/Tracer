"""Data access for the topics and topic_papers tables."""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.models import Paper, Topic, TopicPaper, TopicSynthesis


class TopicRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, query: str, *, user_id: int, field: str | None = None,
                     clarified_query: str | None = None) -> Topic:
        topic = Topic(query=query.strip(), user_id=user_id, field=field, clarified_query=clarified_query)
        self.session.add(topic)
        await self.session.flush()
        return topic

    async def get(self, topic_id: int) -> Topic | None:
        return await self.session.get(Topic, topic_id)

    async def list_recent(self, limit: int = 20, *, user_id: int | None = None) -> list[Topic]:
        """Newest topics first; only one user's topics when user_id is given."""
        query = select(Topic).order_by(Topic.id.desc()).limit(limit)
        if user_id is not None:
            query = query.where(Topic.user_id == user_id)
        return list(await self.session.scalars(query))

    async def summaries(self, user_id: int, limit: int = 50) -> list[dict]:
        """The user's topics, newest first, with their selected-paper count and whether they were analysed.
        ONE query (counts via a join + GROUP BY), not one query per topic."""
        analysed = select(TopicSynthesis.topic_id).where(TopicSynthesis.topic_id == Topic.id).exists()
        query = (select(Topic.id, Topic.query, Topic.field, Topic.created_at, Topic.run_stats.is_not(None),
                        func.count(TopicPaper.rank), analysed)
                 .outerjoin(TopicPaper, (TopicPaper.topic_id == Topic.id) & TopicPaper.rank.is_not(None))
                 .where(Topic.user_id == user_id).group_by(Topic.id).order_by(Topic.id.desc()).limit(limit))
        keys = ("id", "query", "field", "created_at", "has_results", "papers", "analysed")
        return [dict(zip(keys, row)) for row in await self.session.execute(query)]

    async def link_paper(self, topic_id: int, paper_id: int, found_by: list[dict]) -> None:
        """Record that this topic's search found this paper (merging 'found_by' if already linked)."""
        link = await self.session.get(TopicPaper, (topic_id, paper_id))   # look up by the 2-part key
        if link is None:
            self.session.add(TopicPaper(topic_id=topic_id, paper_id=paper_id, found_by=found_by))
        else:
            link.found_by = link.found_by + [f for f in found_by if f not in link.found_by]
        await self.session.flush()

    async def papers_for_topic(self, topic_id: int) -> list[Paper]:
        query = (select(Paper).join(TopicPaper, TopicPaper.paper_id == Paper.id)
                 .where(TopicPaper.topic_id == topic_id).order_by(Paper.id))
        return list(await self.session.scalars(query))

    async def screened(self, topic_id: int) -> list[tuple[TopicPaper, Paper]]:
        """Every candidate the LLM screened for this topic (relevance saved), most relevant first."""
        query = (select(TopicPaper, Paper).join(Paper, Paper.id == TopicPaper.paper_id)
                 .where(TopicPaper.topic_id == topic_id, TopicPaper.relevance.is_not(None))
                 .order_by(TopicPaper.relevance.desc(), Paper.id))
        return [(link, paper) for link, paper in await self.session.execute(query)]

    # ---------- research results (milestone 3) ----------

    async def _link(self, topic_id: int, paper_id: int) -> TopicPaper:
        link = await self.session.get(TopicPaper, (topic_id, paper_id))
        if link is None:
            raise ValueError(f"paper {paper_id} is not linked to topic {topic_id}")
        return link

    async def save_screening(self, topic_id: int, paper_id: int, *, relevance: int, reason: str,
                             study_type: str) -> None:
        link = await self._link(topic_id, paper_id)
        link.relevance, link.relevance_reason, link.study_type = relevance, reason, study_type

    async def save_selection(self, topic_id: int, paper_id: int, *, rank: int, score: float,
                             score_parts: dict, trust: dict, summary: dict | None) -> None:
        link = await self._link(topic_id, paper_id)
        link.rank, link.score, link.score_parts = rank, score, score_parts
        link.trust, link.summary = trust, summary

    async def save_run(self, topic_id: int, *, plan: dict, run_stats: dict) -> None:
        topic = await self.get(topic_id)
        if topic is None:
            raise ValueError(f"topic {topic_id} does not exist")
        topic.plan, topic.run_stats = plan, run_stats
        await self.session.flush()

    async def save_adjusted_trust(self, topic_id: int, paper_id: int, trust: dict) -> None:
        link = await self._link(topic_id, paper_id)
        link.adjusted_trust = trust

    async def save_synthesis(self, topic_id: int, *, connections: list[dict], gaps: dict | None,
                             flags: list[str]) -> None:
        """Store (or replace) a topic's connections and gaps."""
        row = await self.session.get(TopicSynthesis, topic_id)
        if row is None:
            self.session.add(TopicSynthesis(topic_id=topic_id, connections=connections, gaps=gaps, flags=flags))
        else:
            row.connections, row.gaps, row.flags = connections, gaps, flags
        await self.session.flush()

    async def get_synthesis(self, topic_id: int) -> TopicSynthesis | None:
        return await self.session.get(TopicSynthesis, topic_id)

    async def selected(self, topic_id: int) -> list[tuple[TopicPaper, Paper]]:
        """The selected papers of a topic (rank set), best first, with their stored results."""
        query = (select(TopicPaper, Paper).join(Paper, TopicPaper.paper_id == Paper.id)
                 .where(TopicPaper.topic_id == topic_id, TopicPaper.rank.is_not(None))
                 .order_by(TopicPaper.rank))
        return [(link, paper) for link, paper in (await self.session.execute(query)).all()]
