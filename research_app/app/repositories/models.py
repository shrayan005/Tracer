"""
ORM models: each class is one database table.

Tables are created and changed ONLY through Alembic migrations (alembic/versions/),
never with Base.metadata.create_all(), so changes are versioned and safe.
"""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (BigInteger, Date, DateTime, Float, ForeignKey, Index, Integer, Numeric, String, Text,
                        UniqueConstraint, func, text)
from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(),
                                                 onupdate=func.now())


class User(TimestampMixin, Base):
    """A person using the app. The web API needs a login (step 7.3); the CLI acts as the default local user."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True)     # 320 = the longest valid email
    display_name: Mapped[str | None] = mapped_column(String(200))
    password_hash: Mapped[str | None] = mapped_column(String(255))   # argon2; None = can't log in (yet)
    is_active: Mapped[bool] = mapped_column(default=True, server_default="true")   # False = locked out


class Topic(TimestampMixin, Base):
    """One research request typed by a user."""
    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)  # owner
    query: Mapped[str] = mapped_column(Text)
    clarified_query: Mapped[str | None] = mapped_column(Text)
    field: Mapped[str | None] = mapped_column(String(100))   # e.g. "sports science", "computer science"
    # Filled by the research pipeline (milestone 3), so a report can be rebuilt without the LLM:
    plan: Mapped[dict | None] = mapped_column(JSONB)          # the ResearchPlan (brief, queries, flags)
    run_stats: Mapped[dict | None] = mapped_column(JSONB)     # sources, counts, timings, warnings


class Job(TimestampMixin, Base):
    """
    A long run done by the background worker (step 7.1): research a topic, or analyse one.
    The web request only creates this row and queues it; the worker fills in progress and the result.
    status: queued -> running -> done | failed. A failed job is never retried automatically:
    the user starts a new one (research creates a topic, so a blind retry would make a duplicate).
    """
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)   # owner
    kind: Mapped[str] = mapped_column(String(20))                    # research / analyze
    params: Mapped[dict] = mapped_column(JSONB, default=dict)        # {"topic": ...} or {"topic_id": ..}
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    progress: Mapped[list] = mapped_column(JSONB, default=list)      # [{"at": iso time, "message": ...}]
    topic_id: Mapped[int | None] = mapped_column(ForeignKey("topics.id", ondelete="SET NULL"))
    error: Mapped[str | None] = mapped_column(Text)
    retry_of: Mapped[int | None] = mapped_column(ForeignKey("jobs.id", ondelete="SET NULL"))   # a retried job
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        # step 7.4: the DATABASE refuses a second active research for the same user, so two requests in
        # the same instant can't both get through (a check in code alone can be raced)
        Index("uq_jobs_one_active_research", "user_id", unique=True,
              postgresql_where=text("kind = 'research' AND status IN ('queued', 'running')")),
    )


class Paper(TimestampMixin, Base):
    """
    One academic paper, shared by every topic that finds it (the shared cache).
    A paper can be identified by several IDs; each is unique when present.
    """
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(primary_key=True)
    doi: Mapped[str | None] = mapped_column(String(255), unique=True)
    openalex_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    s2_id: Mapped[str | None] = mapped_column(String(64), unique=True)       # Semantic Scholar
    pmid: Mapped[str | None] = mapped_column(String(32), unique=True)        # PubMed
    arxiv_id: Mapped[str | None] = mapped_column(String(64), unique=True)

    title: Mapped[str] = mapped_column(Text)
    authors: Mapped[list] = mapped_column(JSONB, default=list)               # ["Ji, Z.", ...]
    venue: Mapped[str | None] = mapped_column(Text)
    venue_type: Mapped[str | None] = mapped_column(String(50))               # journal / conference / preprint
    publication_type: Mapped[str | None] = mapped_column(String(50))         # article / review / preprint
    pub_types: Mapped[list] = mapped_column(JSONB, default=list, server_default="[]")   # official index tags
    year: Mapped[int | None] = mapped_column(Integer)
    published_date: Mapped[date | None] = mapped_column(Date)
    abstract: Mapped[str | None] = mapped_column(Text)
    tldr: Mapped[str | None] = mapped_column(Text)                           # Semantic Scholar one-liner

    citation_count: Mapped[int | None] = mapped_column(Integer)
    influential_citation_count: Mapped[int | None] = mapped_column(Integer)
    is_open_access: Mapped[bool | None]
    is_retracted: Mapped[bool] = mapped_column(default=False)
    fulltext_status: Mapped[str] = mapped_column(String(20), default="none")  # none / abstract / full
    source_urls: Mapped[dict] = mapped_column(JSONB, default=dict)          # {"pdf": ..., "landing": ...}

    __table_args__ = (Index("ix_papers_year", "year"),)


class TopicPaper(Base):
    """Which papers a topic's search found, and how (source + query). Many-to-many link."""
    __tablename__ = "topic_papers"

    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id", ondelete="CASCADE"), primary_key=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True)
    found_by: Mapped[list] = mapped_column(JSONB, default=list)   # [{"source": ..., "query": ...}]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Results that depend on the TOPIC (the same paper can be relevance 9 for one topic, 2 for another),
    # so they live on the link, not on the paper. All empty until the research pipeline runs.
    relevance: Mapped[int | None] = mapped_column(Integer)          # screening 0-10
    relevance_reason: Mapped[str | None] = mapped_column(Text)
    study_type: Mapped[str | None] = mapped_column(String(30))
    rank: Mapped[int | None] = mapped_column(Integer)               # 1..10 if selected, else NULL
    score: Mapped[float | None] = mapped_column(Float)              # selection score 0-1
    score_parts: Mapped[dict | None] = mapped_column(JSONB)         # {"relevance": 0.45, ...}
    trust: Mapped[dict | None] = mapped_column(JSONB)               # TrustAssessment from design (never changes)
    summary: Mapped[dict | None] = mapped_column(JSONB)             # SummarizedPaper
    adjusted_trust: Mapped[dict | None] = mapped_column(JSONB)      # trust + quality (5.2), always from `trust`


EMBEDDING_DIMENSIONS = 768       # changing it needs a migration (the column type) and a re-index


class PaperChunk(Base):
    """
    One passage of a paper, ready for questions (step 7.6a). Shared by every topic and user that has the
    paper (like analyses): embedded once. Two ways to find it:
      embedding  meaning search (pgvector, HNSW index, cosine distance)
      tsv        keyword search (Postgres full text, GIN index), kept up to date BY THE DATABASE
    index_version = embedding model + dimensions + chunker version: a change re-indexes the paper.
    """
    __tablename__ = "paper_chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer)                  # order in the paper
    kind: Mapped[str] = mapped_column(String(30))                    # abstract / methods / results / table ...
    title: Mapped[str | None] = mapped_column(Text)                  # the section title
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    tsv = mapped_column(TSVECTOR, Computed("to_tsvector('english', text)", persisted=True))
    index_version: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_paper_chunks_embedding", "embedding", postgresql_using="hnsw",
              postgresql_with={"m": 16, "ef_construction": 64}, postgresql_ops={"embedding": "vector_cosine_ops"}),
        Index("ix_paper_chunks_tsv", "tsv", postgresql_using="gin"),
    )


class Chat(TimestampMixin, Base):
    """A conversation about one topic's papers (step 7.6b), owned by one user."""
    __tablename__ = "chats"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(200))              # the first question, shortened


class ChatMessage(Base):
    """One turn: the user's question, or the app's checked answer (stored whole, as JSON)."""
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(10))                # user / assistant
    content: Mapped[str] = mapped_column(Text)                   # the question, or the answer as plain text
    answer: Mapped[dict | None] = mapped_column(JSONB)           # assistant: sentences, citations, flags
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PaperAnalysisRow(TimestampMixin, Base):
    """Claims + appraisal of ONE paper (step 5.1). Topic-independent, so every topic reuses it."""
    __tablename__ = "paper_analyses"

    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True)
    version: Mapped[str] = mapped_column(String(20))             # ANALYSIS_VERSION it was made with
    based_on: Mapped[str] = mapped_column(String(20))            # full text / abstract
    analysis: Mapped[dict] = mapped_column(JSONB)                # AnalyzedPaper


class TopicSynthesis(TimestampMixin, Base):
    """How a topic's selected papers connect, and what the evidence is missing (steps 5.3 + 5.4)."""
    __tablename__ = "topic_syntheses"

    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id", ondelete="CASCADE"), primary_key=True)
    connections: Mapped[list] = mapped_column(JSONB, default=list)    # [Connection]
    gaps: Mapped[dict | None] = mapped_column(JSONB)                  # EvidenceGaps
    flags: Mapped[list] = mapped_column(JSONB, default=list)


class Document(TimestampMixin, Base):
    """
    One stored ORIGINAL file (PDF or XML). The bytes live in file storage under `storage_key`;
    this row knows everything ABOUT the file. Same bytes = same sha256 = one row, one file.
    Visibility: origin "open_access" -> everyone may use it; origin "upload" -> only users
    linked in user_documents.
    """
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    storage_key: Mapped[str] = mapped_column(String(100))
    content_type: Mapped[str] = mapped_column(String(100))          # application/pdf, application/xml
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    origin: Mapped[str] = mapped_column(String(20))                 # open_access / upload
    source_url: Mapped[str | None] = mapped_column(Text)            # where we downloaded it from
    paper_id: Mapped[int | None] = mapped_column(ForeignKey("papers.id", ondelete="SET NULL"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="stored")  # stored / extracted / failed
    page_count: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)                 # why extraction failed


class UserDocument(Base):
    """Which user uploaded which file (private link). Two users, same file -> one file, two links."""
    __tablename__ = "user_documents"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True)
    original_filename: Mapped[str | None] = mapped_column(String(255))   # shown to the user, never a path
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DocumentSection(Base):
    """Extracted text of a document, one row per section, table or caption, in reading order."""
    __tablename__ = "document_sections"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer)                  # 0, 1, 2 ... reading order
    kind: Mapped[str] = mapped_column(String(20))                   # abstract / methods / results / table ...
    title: Mapped[str | None] = mapped_column(Text)                 # heading as written in the paper
    text: Mapped[str] = mapped_column(Text)

    __table_args__ = (UniqueConstraint("document_id", "position"),)


class LlmCall(Base):
    """Every LLM attempt: for cost tracking, debugging and evaluation."""
    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    run_id: Mapped[int | None] = mapped_column(Integer, index=True)   # a job id or an eval run id
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))  # whose budget (7.4)
    __table_args__ = (Index("ix_llm_calls_user_created", "user_id", "created_at"),)   # "tokens used today"
    purpose: Mapped[str] = mapped_column(String(100))                # e.g. "plan", "screen", "health"
    provider: Mapped[str] = mapped_column(String(50))
    model: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(10))                  # ok / error
    attempt: Mapped[int] = mapped_column(Integer)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    thinking_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    error: Mapped[str | None] = mapped_column(Text)
    prompt: Mapped[str | None] = mapped_column(Text)                 # traced purposes only (config)
    response: Mapped[str | None] = mapped_column(Text)
