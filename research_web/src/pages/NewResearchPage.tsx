import { ArrowRight, BookOpen, Filter, Loader2, RotateCcw, Route, Search, Sparkles } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useOutletContext, useSearchParams } from "react-router-dom";
import { ApiError } from "../api/client";
import { isActive, useJob, useJobs, useRetryJob, useStartResearch, useUsage, type PaperCount } from "../api/queries";
import type { JobInfo } from "../api/types";
import { JobSteps } from "../components/JobSteps";
import { jobTitle } from "../lib/format";
import type { LayoutContext } from "../layout/AppLayout";
import { Topbar } from "../layout/Topbar";

const MIN = 3, MAX = 300;

/** Start a research, and follow a run (?job=ID): its steps live in the right panel. */
export function NewResearchPage() {
  const [search] = useSearchParams();
  const jobId = search.get("job") ? Number(search.get("job")) : null;
  const job = useJob(jobId);
  const { setPanel } = useOutletContext<LayoutContext>();

  useEffect(() => { setPanel(job.data ? <JobSteps job={job.data} /> : null); }, [job.data, setPanel]);
  useEffect(() => () => setPanel(null), [setPanel]);
  useOpenTopicWhenDone(job.data);

  return (
    <>
      <Topbar title="New research" />
      <div className="main-body">
        <div className="new-research">
          {job.data ? <RunStatus job={job.data} /> : <StartForm />}
          {job.error && <p className="error">{job.error.message}</p>}
        </div>
      </div>
    </>
  );
}

const EXAMPLES = [
  "creatine supplementation in older adults",
  "caffeine and endurance performance",
  "sleep deprivation and memory",
  "intermittent fasting and weight loss",
];

const how = (n: number) => [
  { icon: Route, title: "Plan", text: "turns your topic into search queries" },
  { icon: Search, title: "Search", text: "academic databases; an agent fills the gaps" },
  { icon: Filter, title: "Screen", text: `AI reads abstracts, keeps the best ${n}` },
  { icon: BookOpen, title: "Read", text: "free full texts where they exist" },
  { icon: Sparkles, title: "Report", text: "summaries, connections, evidence gaps" },
];

/** The choices, with rough time and token use (measured: 10 papers ~2-5 min, ~40-80k tokens). */
const COUNTS: { n: PaperCount; time: string; tokens: string }[] = [
  { n: 5, time: "1–3 min", tokens: "25–45k" },
  { n: 10, time: "2–5 min", tokens: "40–80k" },
  { n: 15, time: "3–8 min", tokens: "60–115k" },
  { n: 20, time: "4–10 min", tokens: "80–150k" },
];

function StartForm() {
  const [topic, setTopic] = useState("");
  const [papers, setPapers] = useState<PaperCount>(10);
  const chosen = COUNTS.find((c) => c.n === papers)!;
  const start = useStartResearch();
  const usage = useUsage();
  const jobs = useJobs();
  const navigate = useNavigate();
  const box = useRef<HTMLTextAreaElement>(null);
  const running = (jobs.data ?? []).find((j) => j.kind === "research" && isActive(j));
  const text = topic.trim();
  const canStart = text.length >= MIN && !start.isPending && !running;

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!canStart) return;
    start.mutate({ topic: text, papers }, { onSuccess: (job) => navigate(`/app/new?job=${job.id}`) });
  };
  const pick = (example: string) => { setTopic(example); box.current?.focus(); };
  const left = usage.data?.remaining ?? null;
  const low = left !== null && usage.data !== undefined && left < usage.data.daily_budget * 0.1;

  return (
    <form className="start" onSubmit={submit}>
      <div className="start-head">
        <h1>What do you want to research?</h1>
        <p className="muted">Tracer finds the studies, reads them, and traces every claim back to its passage.</p>
      </div>

      {running && (
        <div className="notice">
          <Loader2 size={15} className="spin" />
          <span>A research is running: <strong>{jobTitle(running)}</strong>. One at a time.</span>
          <Link className="notice-link" to={`/app/new?job=${running.id}`}>Follow it <ArrowRight size={13} /></Link>
        </div>
      )}

      <div className="composer start-composer">
        <textarea ref={box} rows={2} maxLength={MAX} autoFocus value={topic} aria-label="Research topic"
                  placeholder="e.g. creatine supplementation and muscle strength in older adults"
                  onChange={(e) => setTopic(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) submit(e); }} />
        <button className="start-button" disabled={!canStart}>
          {start.isPending ? <><Loader2 size={15} className="spin" /> Starting…</> : <>Start research <ArrowRight size={15} /></>}
        </button>
      </div>
      <div className="start-meta">
        <span>Enter to start · Shift+Enter for a new line</span>
        <span className={topic.length > MAX - 30 ? "warn" : ""}>{topic.length}/{MAX}</span>
      </div>
      {start.error && <StartError error={start.error} runningId={running?.id} />}

      <div className="paper-count">
        <span className="paper-count-label" id="paper-count-label">Papers to read</span>
        <div className="segmented" role="radiogroup" aria-labelledby="paper-count-label">
          {COUNTS.map((c) => (
            <button type="button" key={c.n} role="radio" aria-checked={papers === c.n}
                    className={papers === c.n ? "seg active" : "seg"} onClick={() => setPapers(c.n)}>
              {c.n}
            </button>
          ))}
        </div>
        <span className="muted paper-count-cost">~{chosen.time} · about {chosen.tokens} tokens</span>
      </div>
      <p className="muted paper-count-help">
        Tracer picks the best {papers} papers, reads their free full text where it exists, and prepares them so you
        can ask questions about them. More papers cover more evidence but take longer and use more of your budget.
      </p>

      <div className="examples">
        <span className="muted">Try:</span>
        {EXAMPLES.map((e) => <button type="button" key={e} className="chip" onClick={() => pick(e)}>{e}</button>)}
      </div>

      <section className="how">
        <p className="how-title">How it works <span className="muted">· about {chosen.time} for {papers} papers, you can leave this page</span></p>
        <ol className="how-steps">
          {how(papers).map(({ icon: Icon, title, text: what }) => (
            <li key={title}>
              <span className="icon-sq brand"><Icon size={15} /></span>
              <strong>{title}</strong>
              <span className="muted">{what}</span>
            </li>
          ))}
        </ol>
        <p className="muted how-after">The same run then prepares the papers for questions, so you can chat with them when it is done.</p>
      </section>

      {usage.data && (
        <p className={`muted usage start-usage${low ? " warn" : ""}`}>
          Today {usage.data.tokens_today.toLocaleString()} tokens used
          {left !== null ? ` · ${left.toLocaleString()} of ${usage.data.daily_budget.toLocaleString()} left` : ""}
          {` · this research uses about ${chosen.tokens}`}
        </p>
      )}
    </form>
  );
}

function StartError({ error, runningId }: { error: Error; runningId?: number }) {
  if (error instanceof ApiError && error.status === 429) {
    return <p className="error">{error.message}</p>;              // "budget used up: resets at 00:00 UTC"
  }
  if (error instanceof ApiError && error.status === 409 && runningId) {
    return <p className="error">{error.message} · <Link className="link-button" to={`/app/new?job=${runningId}`}>follow it</Link></p>;
  }
  return <p className="error">{error.message}</p>;
}

function RunStatus({ job }: { job: JobInfo }) {
  const retry = useRetryJob();
  const navigate = useNavigate();
  const title = jobTitle(job);
  return (
    <div>
      <h2 className="run-title">{title}</h2>
      {isActive(job) && (
        <p className="muted">Running in the background: the steps appear on the right as they happen. You can leave this page;
          the run keeps going and shows in the sidebar.</p>
      )}
      {job.status === "done" && job.topic_id && (
        <p><Link className="small-button inline-flex" to={`/app/topics/${job.topic_id}`}>Open the topic <ArrowRight size={14} /></Link></p>
      )}
      {job.status === "failed" && (
        <>
          <p className="error">This run failed: {job.error}</p>
          {retry.error && <p className="error">{retry.error.message}</p>}
          <button className="small-button inline-flex" disabled={retry.isPending}
                  onClick={() => retry.mutate(job.id, { onSuccess: (next) => navigate(`/app/new?job=${next.id}`) })}>
            <RotateCcw size={14} /> {retry.isPending ? "Starting…" : "Try again"}
          </button>
        </>
      )}
      <p className="muted" style={{ marginTop: 24 }}><Link className="link-button" to="/app/new">Start another research</Link></p>
    </div>
  );
}

/** If you are watching a run when it finishes, open its topic (only on that transition, not on revisits). */
function useOpenTopicWhenDone(job: JobInfo | undefined) {
  const navigate = useNavigate();
  const wasActive = useRef(false);
  useEffect(() => {
    if (!job) return;
    if (wasActive.current && job.status === "done" && job.topic_id) navigate(`/app/topics/${job.topic_id}`);
    wasActive.current = isActive(job);
  }, [job, navigate]);
}
