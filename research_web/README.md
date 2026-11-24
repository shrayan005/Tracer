# research_web

The web app for the research assistant: a plain, fast React UI on top of the API in `../research_app`.

What it does: log in · **New research** (live steps while it runs) · **Library** of topics · a topic opens as a
tab with **Papers** (sortable table, paper details in the right panel), **Connections**, **Gaps**, **Overview**
(+ Markdown download) and **Ask** (questions answered only from the topic's papers, citation chips open the
paper, evidence quotes underneath, ⚠ where a check failed) · **Chats** (saved per topic) · **Search** (passages
across all your papers, no LLM) · **Runs** (background jobs, retry when failed).

## Run it (WSL, three SEPARATE terminals: each command keeps running)

```bash
# terminal 1: the API
cd /mnt/c/Users/Dell/Desktop/LLM/research_app && source ~/.venvs/research_app/bin/activate
uvicorn app.main:app --port 8010

# terminal 2: the worker (research and analyze jobs)
cd /mnt/c/Users/Dell/Desktop/LLM/research_app && source ~/.venvs/research_app/bin/activate
arq app.worker.WorkerSettings

# terminal 3: the web app -> http://localhost:5173
cd /mnt/c/Users/Dell/Desktop/LLM/research_web
npm run dev                  # (the first time: npm install)
```

Stop any of them with Ctrl+C.

The dev server forwards `/api` to the API on port 8010, so the browser sees one origin: the login cookie
works without any CORS setup.

First time: give your CLI user a password so you can log in and see the topics you made with the CLI:

```bash
cd ../research_app && python -m app.cli users set-password local@localhost
```

## Commands

| Command | What |
|---|---|
| `npm run dev` | dev server with live reload |
| `npm run build` | type-check + production build into `dist/` |
| `npm run typecheck` | type-check only |
| `npm test` | unit tests of the helpers (Vitest, ~2 s, no browser) |
| `npm run gen:api` | regenerate `src/api/schema.d.ts` from `openapi.json` |

When the backend's API changes, refresh `openapi.json` and the types:

```bash
cd ../research_app && python -c "import json; from app.main import app; print(json.dumps(app.openapi(), indent=1))" > ../research_web/openapi.json
cd ../research_web && npm run gen:api && npm run typecheck    # shows every place the UI must follow
```

## Layout

```
src/
  api/          client.ts (fetch + errors) · queries.ts (TanStack Query hooks) · types.ts · schema.d.ts (generated)
  layout/       AppLayout (3 panes + login guard) · Sidebar (Topics, Chats, Runs) · Topbar · ErrorBoundary
  pages/        Login · Library · Topic (tabs, 5 views, paper panel) · NewResearch · Search · NotFound
  components/   TopicTabs · PaperTable · PaperPanel · ReportViews · AskView · JobSteps · TrustDots · CopyButton
  lib/          format.ts (+ format.test.ts): names, labels, sorting · openTopics.ts: open tabs, remembered
  styles/       app.css (the only stylesheet: CSS variables, no UI kit)
```

Choices: Vite + React + TypeScript, TanStack Query (cache, prefetch on hover, polling only while a job
runs), React Router, plain CSS, lucide icons. No UI framework, so the bundle stays small (~70 KB gzipped).
