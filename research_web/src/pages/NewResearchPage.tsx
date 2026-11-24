import { ArrowRight, RotateCcw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useOutletContext, useSearchParams } from "react-router-dom";
import { ApiError } from "../api/client";
import { isActive, useJob, useJobs, useRetryJob, useStartResearch, useUsage } from "../api/queries";
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

function StartForm() {
  const [topic, setTopic] = useState("");
  const start = useStartResearch();
  const usage = useUsage();
  const jobs = useJobs();
  const navigate = useNavigate();
  const running = (jobs.data ?? []).find((j) => j.kind === "research" && isActive(j));
  const text = topic.trim();

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    start.mutate(text, { onSuccess: (job) => navigate(`/app/new?job=${job.id}`) });
  };

  return (
    <form onSubmit={submit}>
      <label className="field">
        <span>What do you want to research?</span>
        <textarea className="input topic-input" rows={3} maxLength={MAX} autoFocus value={topic}
                  placeholder="e.g. creatine supplementation and muscle strength in older adults"
                  onChange={(e) => setTopic(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && text.length >= MIN) submit(e); }} />
      </label>
      <p className="muted hint">
        The app plans a search, finds and screens papers in academic databases, picks the best 10 (preferring
        papers with a free full text), reads those full texts and summarises them. It takes 1–3 minutes; you can leave this page.
      </p>
      {start.error && <StartError error={start.error} runningId={running?.id} />}
      <div className="form-row">
        <button className="button button-auto" disabled={text.length < MIN || start.isPending || !!running}>
          {start.isPending ? "Starting…" : "Start research"}
        </button>
        {running && (
          <span className="muted">One research at a time: <Link className="link-button" to={`/app/new?job=${running.id}`}>follow the running one</Link></span>
        )}
      </div>
      {usage.data && (
        <p className="muted usage">
          Today: {usage.data.tokens_today.toLocaleString()} tokens used
          {usage.data.remaining !== null && usage.data.remaining !== undefined
            ? ` · ${usage.data.remaining.toLocaleString()} of ${usage.data.daily_budget.toLocaleString()} left` : ""}
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
