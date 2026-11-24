import { Check, CircleAlert, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import type { JobInfo } from "../api/types";
import { jobTitle } from "../lib/format";

/** The steps of a run, as they happen (like "Searched PubMed → Importing to library…" in a plain list).
 *  Each message is one step; the last one spins while the job runs. */
export function JobSteps({ job }: { job: JobInfo }) {
  const running = job.status === "queued" || job.status === "running";
  const steps = job.progress ?? [];
  return (
    <div className="job-steps">
      <div className="request">
        {jobTitle(job)}
      </div>
      <ol className="steps">
        {job.status === "queued" && <li className="step current"><Loader2 size={14} className="spin" /> Waiting for the worker…</li>}
        {steps.map((s, i) => {
          const current = running && i === steps.length - 1;
          return (
            <li key={i} className={current ? "step current" : "step"}>
              {current ? <Loader2 size={14} className="spin" /> : <Check size={14} />}
              <span>{s.message}</span>
              <span className="step-time">{s.at.slice(11, 19)}</span>
            </li>
          );
        })}
        {job.status === "failed" && (
          <li className="step failed"><CircleAlert size={14} /> <span>{job.error}</span></li>
        )}
      </ol>
      <p className="muted job-meta">
        {statusText(job)} · <Elapsed job={job} />
        {job.tokens ? ` · ${job.tokens.toLocaleString()} tokens` : ""}
      </p>
    </div>
  );
}

function statusText(job: JobInfo) {
  return { queued: "Queued", running: "Running", done: "Done", failed: "Failed" }[job.status];
}

/** Time since the start (ticks every second while running; fixed once finished). */
function Elapsed({ job }: { job: JobInfo }) {
  const start = job.started_at ?? job.created_at;
  const end = job.finished_at;
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (end) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [end]);
  const seconds = Math.max(0, Math.round(((end ? Date.parse(end) : now) - Date.parse(start)) / 1000));
  return <>{seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`}</>;
}
