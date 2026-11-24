"""Shared data shapes (Pydantic) that move between layers."""

from datetime import datetime, date
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------- LLM ----------

class LlmResult(BaseModel):
    """What the LLM gateway returns for one successful call."""
    text: str
    provider: str
    model: str
    finish_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0          # visible answer
    thinking_tokens: int = 0        # hidden reasoning (billed as output)
    latency_s: float = 0.0
    cost_usd: float | None = None   # None when the price is unknown
    attempts: int = 1               # total attempts across retries and fallbacks


class LlmCallRecord(BaseModel):
    """One attempt against one provider, successful or not. Saved to the llm_calls table."""
    purpose: str
    provider: str
    model: str
    status: Literal["ok", "error"]
    attempt: int
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float | None = None
    error: str | None = None
    run_id: int | None = None
    user_id: int | None = None          # whose budget it counts against (7.4); None = admin / evals
    created_at: datetime = Field(default_factory=datetime.now)
    prompt: str | None = None           # only for traced purposes (LLM_TRACE_PURPOSES)
    response: str | None = None


class JobStepUsage(BaseModel):
    """One step (purpose) of a job on one provider: how many calls, how many failed, tokens, time."""
    purpose: str
    provider: str
    calls: int
    errors: int
    tokens: int                         # successful calls only (what counts)
    avg_ms: int
    cost_usd: float = 0.0
    last_error: str | None = None


class LlmTrace(BaseModel):
    """One traced LLM call: exactly what the model was sent and what it answered."""
    id: int
    created_at: datetime
    purpose: str
    provider: str
    model: str
    tokens: int
    latency_ms: int
    prompt: str
    response: str | None = None


class JobDetails(BaseModel):
    """What happened inside a job: every step's LLM use, plus the traced prompts and answers."""
    job_id: int
    steps: list[JobStepUsage]
    traces: list[LlmTrace]


# ---------- Health ----------

class ComponentHealth(BaseModel):
    name: str
    ok: bool
    detail: str = ""


class HealthReport(BaseModel):
    status: Literal["ok", "degraded"]
    components: list[ComponentHealth]



# ---------- Papers ----------

class PaperRecord(BaseModel):
    """
    One paper as returned by ONE source, converted to a common shape.
    Every source client returns these, so services never see API-specific formats.
    """
    source: str                                  # openalex / europepmc / arxiv / semantic_scholar
    title: str

    # identifiers: a paper can have several; any one of them can identify it
    doi: str | None = None
    openalex_id: str | None = None
    s2_id: str | None = None                     # Semantic Scholar id
    pmid: str | None = None                      # PubMed id
    arxiv_id: str | None = None

    # descriptive information
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None                     # journal / conference name
    venue_type: str | None = None                # journal / conference / repository / preprint
    publication_type: str | None = None          # article / review / preprint
    pub_types: list[str] = Field(default_factory=list)   # official index tags, lowercase (PubMed via Europe PMC,
                                                         # Semantic Scholar), e.g. "randomized controlled trial"
    year: int | None = None
    published_date: date | None = None
    abstract: str | None = None
    tldr: str | None = None                      # Semantic Scholar's one-line summary

    # signals we'll use later for ranking and trust
    citation_count: int | None = None
    influential_citation_count: int | None = None
    is_open_access: bool | None = None
    is_retracted: bool = False

    # links
    pdf_url: str | None = None
    landing_url: str | None = None

    found_by_query: str | None = None            # which search query found it


class FoundBy(BaseModel):
    """Which source and which query found a paper."""
    source: str
    query: str


class Candidate(BaseModel):
    """One unique paper after merging duplicates from all sources and queries."""
    paper: PaperRecord
    found_by: list[FoundBy] = Field(default_factory=list)



class ResearchPlan(BaseModel):
    """The LLM's plan for searching a topic (planning step)."""
    field: str = Field(description="main academic field, e.g. 'sports science', 'computer science', 'economics'")
    is_biomedical: bool = Field(description="true if the topic belongs to medicine, biology, health, "
                                            "physiology or sports science")
    is_cs_math_physics: bool = Field(description="true if the topic belongs to computer science, AI, "
                                                 "mathematics, statistics or physics")
    research_brief: str = Field(description="one or two sentences: what the user wants to learn")
    queries: list[str] = Field(min_length=1, max_length=6,
                               description="3 to 5 short search queries (3-8 words each) covering "
                                           "different angles; use technical terms and synonyms")


class SearchReview(BaseModel):
    """The search agent's decision after looking at one round of search results."""
    gaps: str = Field(description="one sentence: which kinds of relevant studies seem to be missing")
    done: bool = Field(description="true if more searching would most likely find nothing new")
    # no max_length here: an LLM that suggests 4 queries should not trigger a retry; code keeps 3
    new_queries: list[str] = Field(default_factory=list,
                                   description="0 to 3 NEW short search queries (2-5 words) aimed at the gaps")


class DiscoveryResult(BaseModel):
    """What the discovery step returns to the caller (CLI / API)."""
    topic_id: int
    plan: ResearchPlan
    sources_used: list[str]
    raw_results: int                              # before removing duplicates
    unique_papers: int                            # after removing duplicates
    per_source: dict[str, int]                    # results per source
    errors: list[str]                             # sources/queries that failed (not fatal)
    candidates: list[Candidate]
    paper_ids: list[int] = Field(default_factory=list)   # database id of each candidate (same order)
    agent_queries: list[str] = Field(default_factory=list)   # extra queries the search agent ran
    agent_rounds: list[dict] = Field(default_factory=list)   # the agent's decision per extra round

# ---------- Selection ----------

StudyType = Literal[
    "meta_analysis", "systematic_review", "narrative_review", "randomized_trial", "cohort",
    "case_control", "cross_sectional", "case_report", "lab_experiment", "computational",
    "theoretical", "other",
]


class PaperScreening(BaseModel):
    """The LLM's judgement of ONE paper's abstract."""
    id: int = Field(description="the paper's id exactly as given in the input")
    relevance: int = Field(ge=0, le=10, description="0 = unrelated, 5 = partly relevant, 10 = exactly on topic")
    reason: str = Field(description="one short sentence explaining the relevance score")
    study_type: StudyType = Field(description="the study design, judged from the title and abstract")


class ScreeningBatch(BaseModel):
    """What the LLM returns for one batch of papers."""
    results: list[PaperScreening]



TrustLabel = Literal["high", "moderate", "low", "very low", "retracted"]
FieldFamily = Literal["biomedical", "sport_biomechanics", "cs", "physics_math", "general"]


class TrustAssessment(BaseModel):
    """How much to trust a paper as evidence: a level, a label, and the reasons (never a mystery number)."""
    level: int                 # 0 (retracted) .. 5 (strongest)
    label: TrustLabel
    reasons: list[str]


class Finding(BaseModel):
    """One key result, with the passage from the paper that supports it (step 4.5)."""
    statement: str = Field(description="one key result, in plain words")
    quote: str | None = Field(default=None, description="exact passage copied from the paper text that supports it")


class PaperSummary(BaseModel):
    """What the LLM returns for ONE paper (from the full text if we have it, else the abstract)."""
    tldr: str = Field(description="one plain-English sentence with the main result")
    objective: str = Field(description="what the study set out to find")
    methods: str = Field(description="design, participants (how many, who), what was measured and how")
    key_findings: list[Finding] = Field(min_length=1, max_length=5)
    limitations: list[str] = Field(default_factory=list,
                                   description="limitations the authors themselves state; empty if none")
    why_relevant: str = Field(description="one sentence on how the paper relates to the research brief")

    @field_validator("key_findings", mode="before")
    @classmethod
    def _old_format(cls, value):
        """Summaries saved before step 4.5 stored findings as plain strings: keep them readable."""
        return [{"statement": v} if isinstance(v, str) else v for v in value or []]


class SummarizedPaper(BaseModel):
    """A summary plus what CODE knows about it: what it was based on, and any warnings."""
    summary: PaperSummary | None             # None when there was nothing to summarise
    based_on: Literal["full text", "abstract", "title only"]
    flags: list[str] = Field(default_factory=list)


# ---------- Research report (step 3.6) ----------

class ReportPaper(BaseModel):
    """One selected paper with everything the report shows about it."""
    rank: int
    paper_id: int
    paper: PaperRecord
    relevance: int
    relevance_reason: str
    study_type: StudyType
    trust: TrustAssessment                   # quality-adjusted (5.2) if analysed, else the design-based trust
    score: float
    score_parts: dict[str, float]
    summary: SummarizedPaper | None          # None if the summary call failed
    apa: str
    in_text: str
    bibtex_key: str
    bibtex: str
    base_trust: TrustAssessment | None = None        # design-only trust, when quality changed it (milestone 5)
    analysis: "AnalyzedPaper | None" = None           # claims + appraisal (milestone 5)


class SearchAgentRound(BaseModel):
    """One extra search round the search agent decided on (saved in the topic's run_stats)."""
    round: int
    gaps: str
    done: bool
    new_queries: list[str] = Field(default_factory=list)
    new_papers: int = 0


class ResearchReport(BaseModel):
    """The result of the research pipeline, rebuilt from the database (so it is always reproducible)."""
    topic_id: int
    topic: str
    created_at: datetime | None = None
    plan: ResearchPlan
    field_family: FieldFamily
    sources_used: list[str]
    counts: dict[str, int]                   # found / prefiltered / screened / selected
    timings_s: dict[str, float]              # seconds per stage
    warnings: list[str]
    papers: list[ReportPaper]
    connections: list["Connection"] = Field(default_factory=list)      # milestone 5 (old reports: empty)
    gaps: "EvidenceGaps | None" = None
    synthesis_flags: list[str] = Field(default_factory=list)
    search_agent: list[SearchAgentRound] = Field(default_factory=list)   # old reports / agent off: empty


# ---------- Full text (milestone 4) ----------

class FullTextLink(BaseModel):
    """One place where a legal free full text of a paper might be (step 4.2 finds, 4.3 downloads)."""
    kind: Literal["xml", "pdf"]          # xml = structured (JATS), pdf = needs text extraction
    url: str
    via: str                             # europepmc / arxiv / metadata / unpaywall
    license: str | None = None           # e.g. "cc-by", when the source says


class FetchResult(BaseModel):
    """What happened when we tried to get one paper's full text (step 4.3)."""
    paper_id: int
    document_id: int | None = None       # None: no free full text could be downloaded
    kind: Literal["xml", "pdf"] | None = None
    via: str | None = None               # which link worked: europepmc / arxiv / metadata / unpaywall
    url: str | None = None
    size_bytes: int | None = None
    reused: bool = False                 # True: already stored earlier, nothing downloaded
    failures: list[str] = Field(default_factory=list)   # why each tried link failed


class ExtractionResult(BaseModel):
    """What happened when a stored file was turned into sections (step 4.4)."""
    document_id: int
    status: Literal["extracted", "failed"]
    sections: int = 0
    characters: int = 0
    kinds: list[str] = Field(default_factory=list)     # e.g. ["introduction", "methods", "results"]
    error: str | None = None
    reused: bool = False                               # True: extracted earlier, nothing re-parsed


# ---------- Paper analysis (milestone 5) ----------

Direction = Literal["increase", "decrease", "no difference", "association", "other"]
FactValue = Literal["yes", "no", "not reported", "not applicable"]


class Claim(BaseModel):
    """One result of a study, written so it can be compared with other papers' results."""
    outcome: str = Field(description="what was measured, e.g. 'long head EMG amplitude'")
    factor: str = Field(description="what was varied or compared, e.g. 'narrow vs wide grip' or "
                                    "'training (before vs after)'")
    population: str = Field(description="who or what was studied, e.g. '10 healthy young men'")
    direction: Direction = Field(description="how the outcome changed with the factor")
    magnitude: str | None = Field(default=None, description="size as reported, e.g. '+18.5%', 'p < 0.05'")
    quote: str = Field(description="exact passage from the paper that states this result")


class Fact(BaseModel):
    value: FactValue
    quote: str | None = Field(default=None, description="exact passage that shows it (needed for 'yes')")


class Appraisal(BaseModel):
    """How the study was done: the facts a quality checklist asks for (scored by code in 5.2)."""
    sample_size: int | None = Field(description="number of participants/subjects analysed; null if not applicable")
    sample_size_quote: str | None = None
    sex: Literal["male", "female", "mixed", "not reported", "not applicable"]
    duration: Literal["single session", "under 3 months", "3 months or more", "not applicable", "not reported"]
    control_group: Fact = Field(description="a control or comparison group/condition")
    randomized: Fact = Field(description="participants or conditions randomly assigned/ordered")
    blinded: Fact = Field(description="participants or assessors blinded")
    statistics_reported: Fact = Field(description="statistical tests with p-values or confidence intervals")
    effect_sizes: Fact = Field(description="effect sizes reported (e.g. Cohen's d, eta squared)")
    conflicts_declared: Fact = Field(description="a conflict-of-interest / competing-interests statement")
    data_available: Fact = Field(description="data or code availability statement")


class PaperAnalysis(BaseModel):
    """What the LLM returns for ONE paper (step 5.1)."""
    claims: list[Claim] = Field(min_length=1, max_length=6)
    appraisal: Appraisal


class AnalyzedPaper(BaseModel):
    """The analysis plus what CODE found when checking it."""
    analysis: PaperAnalysis
    based_on: Literal["full text", "abstract"]
    flags: list[str] = Field(default_factory=list)


# ---------- Connections between papers (step 5.3) ----------

Stance = Literal["supports", "opposes", "no effect"]
Relation = Literal["agree", "contradict", "mixed"]


class GroupMember(BaseModel):
    claim_id: str = Field(description="id of a claim from the list, e.g. 'P3.C2'")
    stance: Stance = Field(description="how this claim answers the group's question")


class ClaimGroup(BaseModel):
    """What the LLM proposes: claims from different papers that answer one question."""
    question: str = Field(description="ONE yes/no question every member answers (name the factor and the outcome)")
    members: list[GroupMember] = Field(min_length=2)
    finding: str = Field(description="one sentence: what these papers say together")
    why_differ: str | None = Field(default=None, description="if the papers disagree: the most likely reasons, "
                                   "using only differences shown in the data")


class Connections(BaseModel):
    """What the LLM returns for a whole topic (step 5.3)."""
    groups: list[ClaimGroup] = Field(default_factory=list, max_length=8)


class ConnectionMember(BaseModel):
    paper: int                      # the caller's key for the paper (e.g. its database id)
    claim: int                      # index of the claim in that paper's analysis (0-based)
    stance: Stance
    claim_snapshot: Claim | None = None     # the claim as it was when connected (survives a re-analysis)


class Connection(BaseModel):
    """A CHECKED group: the relation is computed by code, members point at real papers and claims."""
    question: str
    relation: Relation
    finding: str
    why_differ: str | None = None
    members: list[ConnectionMember]


# ---------- Research gaps (step 5.4) ----------

class GapWriteUp(BaseModel):
    """What the LLM writes from the counted facts (it may not add gaps or numbers)."""
    summary: str = Field(description="2-4 sentences: what the evidence base is missing, using ONLY the facts given")
    next_studies: list[str] = Field(default_factory=list, max_length=4,
                                    description="concrete study ideas that would fill these gaps")


class SharedLimitation(BaseModel):
    theme: str                       # e.g. "small sample"
    papers: list[str]                # in-text citations of the papers whose authors admit it


class EvidenceGaps(BaseModel):
    """What this body of evidence is missing: found by code, optionally written up by the LLM."""
    n_papers: int
    gaps: list[str]
    notes: list[str] = Field(default_factory=list)   # about what THIS APP could read, not gaps in the research
    shared_limitations: list[SharedLimitation] = Field(default_factory=list)
    papers_with_limitations: int = 0
    write_up: GapWriteUp | None = None
    flags: list[str] = Field(default_factory=list)


# ---------- Single-paper detail view (step 5.7) ----------

class SectionInfo(BaseModel):
    kind: str
    title: str | None
    characters: int


class DocumentInfo(BaseModel):
    """The full text a paper was read from (steps 4.3/4.4)."""
    content_type: str                # application/pdf or application/xml
    source_url: str | None
    size_bytes: int
    page_count: int | None
    status: str                      # stored / extracted / failed
    error: str | None = None
    sections: list[SectionInfo] = Field(default_factory=list)


class PaperDetail(BaseModel):
    """Everything known about ONE selected paper of a topic."""
    topic_id: int
    topic: str
    field_family: FieldFamily
    n_papers: int
    paper: ReportPaper
    connections: list[Connection] = Field(default_factory=list)   # only the ones this paper is part of
    labels: dict[int, str] = Field(default_factory=dict)          # paper_id -> in-text citation
    ranks: dict[int, int] = Field(default_factory=dict)           # paper_id -> rank (for links)
    document: DocumentInfo | None = None                          # None: abstract only


# ---------- Questions over papers (step 7.6b) ----------

class AnswerSentence(BaseModel):
    """What the LLM writes: one sentence, the excerpt it comes from, and the words that show it."""
    text: str = Field(description="one sentence of the answer")
    excerpt: str = Field(description="the id of the ONE excerpt this sentence comes from, e.g. 'E3'")
    quote: str = Field(description="words copied EXACTLY from that excerpt that support the sentence (max 30 words)")


class AnswerDraft(BaseModel):
    sentences: list[AnswerSentence] = Field(default_factory=list, max_length=10)
    answered: bool = Field(description="false if the excerpts do not answer the question")
    missing: str | None = Field(default=None, description="what the excerpts do not cover, if anything")


class Citation(BaseModel):
    paper_id: int
    rank: int | None = None          # the paper's rank in the topic (opens it in the UI)
    label: str                       # "(Brose et al., 2003)"
    section: str                     # "results", "abstract", ...
    quote: str
    chunk_id: int
    passage: str = ""               # the whole passage the quote is from (the UI highlights the quote in it)


class AnswerPart(BaseModel):
    text: str
    citation: Citation | None = None
    flags: list[str] = Field(default_factory=list)     # what code found wrong ("quote not found in the paper")


class Answer(BaseModel):
    """A CHECKED answer: every sentence points at a passage, and code checked its quote and numbers."""
    parts: list[AnswerPart] = Field(default_factory=list)
    answered: bool
    missing: str | None = None
    passages_searched: int = 0
    flags: list[str] = Field(default_factory=list)     # about the whole answer (e.g. papers not prepared)


class ChatMessageInfo(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    answer: Answer | None = None
    created_at: datetime


class ChatInfo(BaseModel):
    id: int
    topic_id: int
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessageInfo] = Field(default_factory=list)


class AskResult(BaseModel):
    chat_id: int
    question: ChatMessageInfo
    answer: ChatMessageInfo


class Readiness(BaseModel):
    """Is a topic ready for questions? (passages + embeddings, step 7.6a)"""
    papers: int
    prepared: int                    # papers with passages
    passages: int


class SearchHit(BaseModel):
    """A passage found by the Search page (no LLM)."""
    paper_id: int
    topic_id: int
    rank: int
    label: str
    title: str
    section: str
    snippet: str                     # matched words wrapped in \u0002 ... \u0003


# ---------- Users (step 7.3) ----------

class UserInfo(BaseModel):
    """Who is logged in (never the password hash)."""
    id: int
    email: str
    display_name: str | None = None


class ProviderUsage(BaseModel):
    """One LLM provider's share of a user's calls today: which provider did the work, and which one fails."""
    provider: str                    # groq / gemini / openrouter
    ok_calls: int
    failed_calls: int                # 429s, timeouts, 5xx... (each attempt counts; the gateway may have recovered)
    tokens: int                      # successful calls, embeddings left out (as in the budget)
    last_error: str | None = None
    last_error_at: datetime | None = None


class UsageInfo(BaseModel):
    """A user's LLM use today (UTC) against their daily budget (step 7.4)."""
    tokens_today: int
    cost_today_usd: float
    daily_budget: int                # 0 = no limit
    remaining: int | None            # None = no limit
    resets_at: datetime              # next midnight UTC
    providers: list[ProviderUsage] = []


# ---------- Topic list (step 7.2) ----------

class TopicSummary(BaseModel):
    """One row of 'my topics'."""
    id: int
    query: str
    field: str | None = None
    created_at: datetime
    has_results: bool                # False: still running, or the research failed
    papers: int                      # selected papers
    analysed: bool                   # connections + gaps done (milestone 5)


# ---------- Background jobs (step 7.1) ----------

JobKind = Literal["research", "analyze", "index"]
JobStatus = Literal["queued", "running", "done", "failed"]


class JobProgress(BaseModel):
    at: str                          # ISO time
    message: str


class JobInfo(BaseModel):
    """What a user sees about one of their jobs (CLI now, API in 7.2)."""
    id: int
    kind: JobKind
    params: dict
    status: JobStatus
    progress: list[JobProgress] = Field(default_factory=list)
    topic_id: int | None = None
    error: str | None = None
    retry_of: int | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    tokens: int = 0                  # every LLM call the job made (llm_calls.run_id = the job id)
    cost_usd: float = 0.0


# ReportPaper / ResearchReport refer to models defined further down this file (by name, in quotes):
# now that every model exists, let pydantic resolve those names.
ReportPaper.model_rebuild()
ResearchReport.model_rebuild()
