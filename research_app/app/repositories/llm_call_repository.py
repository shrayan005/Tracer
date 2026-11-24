"""Data access for the llm_calls table. No business logic here."""

from decimal import Decimal

from sqlalchemy import case, func, select
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.schemas import LlmCallRecord
from app.repositories.models import LlmCall


class LlmCallRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add(self, record: LlmCallRecord) -> LlmCall:
        row = LlmCall(
            run_id=record.run_id, user_id=record.user_id, purpose=record.purpose, provider=record.provider,
            model=record.model, status=record.status, attempt=record.attempt,
            input_tokens=record.input_tokens, output_tokens=record.output_tokens,
            thinking_tokens=record.thinking_tokens, latency_ms=record.latency_ms,
            cost_usd=Decimal(str(record.cost_usd)) if record.cost_usd is not None else None,
            error=record.error, prompt=record.prompt, response=record.response,
        )
        self.session.add(row)          # add() only stages the row in memory: no waiting
        await self.session.flush()     # flush() talks to Postgres (assigns row.id): await
        return row

    async def recent(self, limit: int = 20) -> list[LlmCall]:
        result = await self.session.scalars(select(LlmCall).order_by(LlmCall.id.desc()).limit(limit))
        return list(result)

    async def total_cost(self, run_id: int | None = None) -> float:
        query = select(func.coalesce(func.sum(LlmCall.cost_usd), 0))
        if run_id is not None:
            query = query.where(LlmCall.run_id == run_id)
        return float(await self.session.scalar(query))

    async def steps(self, run_id: int) -> list[dict]:
        """Per purpose and provider, in the order they first ran: calls, errors, tokens (ok calls), avg time."""
        ok = LlmCall.status == "ok"
        tokens = LlmCall.input_tokens + LlmCall.output_tokens + LlmCall.thinking_tokens
        rows = (await self.session.execute(
            select(LlmCall.purpose, LlmCall.provider, func.count(),
                   func.count().filter(LlmCall.status == "error"),
                   func.coalesce(func.sum(case((ok, tokens), else_=0)), 0),
                   func.coalesce(func.avg(LlmCall.latency_ms), 0),
                   func.coalesce(func.sum(LlmCall.cost_usd), 0),
                   func.max(LlmCall.error))
            .where(LlmCall.run_id == run_id)
            .group_by(LlmCall.purpose, LlmCall.provider)
            .order_by(func.min(LlmCall.created_at)))).all()
        return [dict(purpose=p, provider=pr, calls=c, errors=e, tokens=int(t), avg_ms=int(a), cost_usd=float(cost),
                     last_error=err[:300] if e and err else None) for p, pr, c, e, t, a, cost, err in rows]

    async def traces(self, run_id: int) -> list[LlmCall]:
        """The job's calls that kept their prompt (traced purposes), oldest first."""
        return list((await self.session.scalars(
            select(LlmCall).where(LlmCall.run_id == run_id, LlmCall.prompt.is_not(None))
            .order_by(LlmCall.created_at, LlmCall.id))).all())

    async def usage(self, run_id: int) -> tuple[int, float]:
        """(tokens, cost in USD) of every successful call tagged with this run (e.g. a job id)."""
        row = (await self.session.execute(
            select(func.coalesce(func.sum(LlmCall.input_tokens + LlmCall.output_tokens + LlmCall.thinking_tokens), 0),
                   func.coalesce(func.sum(LlmCall.cost_usd), 0))
            .where(LlmCall.run_id == run_id, LlmCall.status == "ok"))).one()
        return int(row[0]), float(row[1])

    async def user_usage_since(self, user_id: int, since) -> tuple[int, float]:
        """(tokens, cost) of a user's successful calls since a moment (e.g. midnight UTC): the daily budget.
        Embeddings are left out (measured 7.6a: indexing one topic ~75k estimated tokens would eat the budget):
        they are done once per paper and shared, and paced by their own limit; the budget protects CHAT models."""
        row = (await self.session.execute(
            select(func.coalesce(func.sum(LlmCall.input_tokens + LlmCall.output_tokens + LlmCall.thinking_tokens), 0),
                   func.coalesce(func.sum(LlmCall.cost_usd), 0))
            .where(LlmCall.user_id == user_id, LlmCall.status == "ok", LlmCall.created_at >= since,
                   LlmCall.purpose != "embed"))).one()
        return int(row[0]), float(row[1])

    async def user_providers_since(self, user_id: int, since) -> list[dict]:
        """Per provider, a user's calls since a moment: {provider, ok_calls, failed_calls, tokens, last_error,
        last_error_at}. Tokens leave out embeddings (same rule as the budget, so they add up to it); the call
        counts include them, because a rate-limited embedding is exactly the free-tier trouble worth seeing."""
        is_ok = LlmCall.status == "ok"
        tokens = case((is_ok & (LlmCall.purpose != "embed"),
                       LlmCall.input_tokens + LlmCall.output_tokens + LlmCall.thinking_tokens), else_=0)
        rows = (await self.session.execute(
            select(LlmCall.provider,
                   func.count().filter(is_ok), func.count().filter(~is_ok), func.coalesce(func.sum(tokens), 0))
            .where(LlmCall.user_id == user_id, LlmCall.created_at >= since)
            .group_by(LlmCall.provider).order_by(LlmCall.provider))).all()
        # the newest failure per provider (Postgres DISTINCT ON); ok calls may carry a note in `error`: skip them
        last = {r.provider: r for r in (await self.session.execute(
            select(LlmCall.provider, LlmCall.error, LlmCall.created_at)
            .where(LlmCall.user_id == user_id, LlmCall.created_at >= since, ~is_ok)
            .ext(distinct_on(LlmCall.provider))   # id breaks ties: now() is the same for every row of one transaction
            .order_by(LlmCall.provider, LlmCall.created_at.desc(), LlmCall.id.desc()))).all()}
        return [{"provider": p, "ok_calls": ok, "failed_calls": failed, "tokens": int(t),
                 "last_error": last[p].error if p in last else None,
                 "last_error_at": last[p].created_at if p in last else None}
                for p, ok, failed, t in rows]
