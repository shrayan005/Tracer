# Research Paper Assistant

Type a topic → find the best 5–10 papers → summarise them (from the free full text where possible,
with quotes as evidence) → judge their quality → connect them (agree / contradict / mixed, and why)
→ find what the evidence is missing → cited report → **ask questions** answered only from those papers,
every sentence citing a passage whose quote code checked. Multi-user web app (login, background jobs,
per-user budgets) with a React UI in `../research_web`, plus a CLI and an evaluation harness.
A layered **modular monolith**: Presentation → Service → Repository / Integration.

## Environment
Code lives here on Windows; everything **runs in WSL (Ubuntu)**.

| Component | Where |
|---|---|
| Code + `.env` | `C:\Users\Dell\Desktop\LLM\research_app` (= `/mnt/c/Users/Dell/Desktop/LLM/research_app`) |
| Python venv | `~/.venvs/research_app` (inside WSL) |
| Postgres 18 + pgvector 0.8 | WSL, **port 5433** (Windows Postgres uses 5432); password in `~/.pgpass` |
| Redis | WSL, port 6379: job queue, login sessions, failed-login counters, "no full text" memory |
| Downloaded full texts | `data/files/<aa>/<bb>/<sha256>.pdf\|xml` (git-ignored, named by fingerprint) |
| Reports | `reports/<topic id>-<topic>.md` (git-ignored) |
| API server | **port 8010** (8000 is taken by another app) · web app (dev): port 5173 |

## Run it (three WSL terminals: each keeps running)
```bash
# 1: the API
cd /mnt/c/Users/Dell/Desktop/LLM/research_app && source ~/.venvs/research_app/bin/activate
WATCHFILES_FORCE_POLLING=true uvicorn app.main:app --port 8010 --reload   # → http://localhost:8010/docs

# 2: the worker (research, analyse and "prepare for questions" jobs)
cd /mnt/c/Users/Dell/Desktop/LLM/research_app && source ~/.venvs/research_app/bin/activate
arq app.worker.WorkerSettings

# 3: the web app → http://localhost:5173  (see ../research_web/README.md)
cd /mnt/c/Users/Dell/Desktop/LLM/research_web && npm run dev
```
First time: `python -m app.cli users set-password local@localhost` lets you log in to the web app as the
user who owns the topics made with the CLI.

## Command line
```bash
python -m app.cli health --deep          # check DB, Redis, LLM
python -m app.cli research "topic"       # full pipeline in this terminal → reports/<id>-<topic>.md
python -m app.cli report 57              # rebuild topic 57's report from the DB (no LLM calls)
python -m app.cli fetch 57 [--retry]     # download + extract free full texts of topic 57's papers
python -m app.cli analyze 57 [--refresh] # claims, quality, connections, gaps
python -m app.cli index 57 [--refresh]   # prepare topic 57's papers for questions (passages + embeddings)
python -m app.cli paper 57 3             # detailed page for the paper ranked 3
python -m app.cli jobs submit research "topic" | analyze 57 | index 57   # the same, as background jobs
python -m app.cli jobs status 42 · jobs list · jobs retry 42 · jobs usage
python -m app.cli users add you@example.com --name You · users set-password <email>
python -m app.cli usage                  # recent LLM calls + total cost

python -m app.evals run <suite>          # evaluation (see app/evals/__main__.py): smoke, stability, grounding,
                                         #   judge-check, appraisal, screening, search-recall, connections, ask

python -m pytest                         # unit tests (no LLM tokens, no network)
python -m pytest -m integration          # real Postgres + Redis
alembic revision --autogenerate -m "…" && alembic upgrade head    # after editing repositories/models.py
```

## The pipeline
```
plan (LLM) → search (OpenAlex, Europe PMC, arXiv, Semantic Scholar) → merge + save
→ prefilter (keywords) → screen (LLM, 20 at a time, stops early) → study type checked against index tags
→ trust (field rules) → select (score + MMR diversity) → full text (find → safe download → extract)
→ summarise (LLM, a quote per finding, checked) → analyse (claims + appraisal, every quote checked, per paper)
→ quality (code) → connect (LLM groups; CODE decides agree/contradict/mixed) → gaps (CODE counts)
→ prepare for questions (passages ~1,000 chars → Gemini embeddings → pgvector) → save → report

ask:  question → meaning search (pgvector) + keyword search (Postgres full text) → merged (RRF), ≤3 per paper
      → LLM answers from 8 passages only → CODE checks each sentence's passage, quote and numbers → saved chat
```
A research job takes ~1.5 min plus ~3–4 min the first time its papers are prepared for questions
(Gemini's free tier embeds ~90 passages a minute). Everything per paper is stored once and reused by
every topic and user.

## Web API (all need a login except health and auth)
| | |
|---|---|
| `POST /api/auth/register · login · logout`, `GET /api/auth/me` | email + password; session in an httpOnly cookie |
| `POST /api/research` → job · `GET /api/jobs[/{id}]` · `POST /api/jobs/{id}/retry` · `GET /api/usage` | background work, live progress, daily budget |
| `GET /api/topics[/{id}]` · `/report.md` · `/papers/{rank}` · `POST /api/topics/{id}/analyze` | results |
| `POST /api/topics/{id}/ask` · `GET /readiness` · `POST /prepare` · `GET/DELETE /api/chats[/{id}]` | questions + chats |
| `GET /api/search?q=` | passages across your papers (no LLM) |

Errors are the same everywhere: 401 not logged in · 404 missing **or another user's** (indistinguishable) ·
409 not ready / conflict · 429 too many logins or daily budget used up (with `Retry-After`).

## Useful settings (`.env`)
| Setting | Default | What it does |
|---|---|---|
| `LLM_PROVIDER_ORDER` / `LLM_ROUTES` | groq → gemini → openrouter; summarize/analyze/gaps/chat → gemini, connect → groq | fallback order, per-task routes |
| `LLM_SWITCH_AFTER_SECONDS` / `LLM_WAIT_FOR` | 5 / `{"connect": 60}` | switch provider instead of waiting, except tasks whose quality depends on their model |
| `PREFILTER_KEEP` / `SCREEN_FIRST` / `SELECT_MAX` | 40 / 20 / 10 | how many papers are screened and selected |
| `FULLTEXT_SHORTLIST` | 12 | best papers whose free full text is tried; the final pick prefers the ones it could read |
| `FULLTEXT_ENABLED` · `ANALYSIS_ENABLED` · `INDEX_ENABLED` | true | the optional pipeline stages |
| `UNPAYWALL_EMAIL` | – | required to use Unpaywall (finds legal free copies) |
| `ENABLED_SOURCES` / `EUROPEPMC_FULLTEXT` | openalex, arxiv, semantic_scholar / false | search sources; Europe PMC is off (often down) |
| `EMBEDDING_MODEL` | gemini-embedding-001 (768 dims, fixed by the DB column) | embeddings for questions |
| `DAILY_TOKEN_BUDGET` | 250000 | LLM tokens per user per UTC day (embeddings not counted); 0 = no limit |
| `WORKER_MAX_JOBS` / `JOB_TIMEOUT_MINUTES` | 2 / 15 | the worker |
| `REGISTRATION_OPEN` / `COOKIE_SECURE` / `SESSION_DAYS` | true / false / 14 | login (set `COOKIE_SECURE=true` behind HTTPS) |

When the analysis prompt changes, raise `ANALYSIS_VERSION`; when chunking changes, `CHUNKER_VERSION`:
stored results made with another version are redone automatically when next needed.

## Layout
```
app/
  presentation/   api/ (auth, jobs, topics, ask, health controllers), report_markdown.py, paper_markdown.py
  services/       research pipeline (discovery … synthesis), jobs, auth, chunking + index (passages),
                  retrieval (meaning + keyword search), ask (checked answers, chats), study_type, grounding
  repositories/   models.py (tables), session.py, *_repository.py
  integrations/   llm/ (providers, gateway, run/user context), embeddings/ (Gemini, paced), sources/, storage/,
                  extraction/, downloader.py, miss_cache.py
  evals/          frozen cases, gold labels, suites, scorecards (milestone 6 + 7.7)
  worker.py       arq worker (background jobs)
  config.py · dependencies.py (composition root) · cli.py · main.py
alembic/  evals/gold/ (labels, in git)  evals/cases/ + evals/results/ (git-ignored)  tests/
```

## Rules
- Controllers call services only; services hold business logic; repositories only touch the DB.
- All LLM calls go through `LlmGateway`; every call is logged with its job and its user (`llm_calls`).
- Every topic, job and chat belongs to a user; another user's looks exactly like a missing one.
- Passwords: argon2 only. Sessions: random tokens, Redis stores only their SHA-256, httpOnly + SameSite=Lax.
- Anything downloaded is untrusted: `SafeDownloader` (SSRF guard, size/time limits, type check), `defusedxml`.
- **The LLM extracts, code checks**: quotes and numbers verified against the text; checklist "yes" needs a
  quote; agree/contradict computed from stances; gaps counted by code; answers cite passages whose quotes
  are checked; a failed check is shown (⚠), never hidden.

## Measured quality (milestones 6–7, details in `evals/results/`)
| What | Result | Notes |
|---|---|---|
| Search: trials the reviews used that our search finds | 31% (was 17–24%) | the sources hold ~90%; the planner's queries are the limit |
| Screening: relevant papers kept | 90% | |
| Appraisal facts correct | 93–95% | conflict/data statements and statistics no longer trimmed away |
| Judge agrees with labels | 77% (95% on the clear cases) | |
| Connections: gold groups found | 89–92% on Groq | a fallback model finds far fewer: connect waits for Groq |
| Questions: right paper retrieved / answered with it | 100% / 100% (11 questions) | declines unanswerable ones 100%; ~92–94% of quotes pass |

Gold labels were written by Claude (an AI), marked `labeled_by: claude`, with uncertain ones flagged.

## Limits to know
- **Free tiers are the real capacity**: Groq 200k tokens/day, Gemini embeddings 100 texts/minute (and the
  free tier can be withdrawn for a project: then a paid key is needed). The per-user budget shares them fairly
  but cannot add capacity.
- Full text is often missing (paywalls): 4–6 of 10 papers abstract-only; reports and answers say so.
- Gold sets are small (2 frozen topics, AI-written labels): numbers show direction, not precision.

## Licence note
PDF extraction uses **PyMuPDF (AGPL-3.0)**, only in `app/integrations/extraction/pdf.py`.
**With login (milestone 7) this app can now be used by others over a network: before that happens, replace it
(e.g. GROBID, Apache-2.0) or buy a commercial licence**, otherwise AGPL requires publishing the app's source.
