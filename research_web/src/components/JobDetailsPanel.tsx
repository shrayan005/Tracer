import { ChevronRight } from "lucide-react";
import { useState } from "react";
import { useJobDetails } from "../api/queries";
import type { JobInfo, LlmTrace } from "../api/types";

/** "What happened inside": LLM calls per step (errors, tokens, time) and, for traced steps such as the
 *  search agent, the exact prompt and answer. Closed by default; loads only when opened. */
export function JobDetailsPanel({ job }: { job: JobInfo }) {
  const [open, setOpen] = useState(false);
  const running = job.status === "queued" || job.status === "running";
  const details = useJobDetails(job.id, open, running);

  return (
    <details className="job-details" onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary><ChevronRight size={14} className="chev" /> Details: AI calls per step</summary>
      {details.isPending && open && <p className="muted">Loading…</p>}
      {details.error && <p className="error">{details.error.message}</p>}
      {details.data && (
        <>
          {details.data.steps.length === 0 ? (
            <p className="muted">No AI calls yet.</p>
          ) : (
            <div className="table-scroll">
              <table className="calls-table">
                <thead>
                  <tr><th>Step</th><th>Provider</th><th className="num">Calls</th><th className="num">Failed</th>
                    <th className="num">Tokens</th><th className="num">Avg time</th></tr>
                </thead>
                <tbody>
                  {details.data.steps.map((s) => (
                    <tr key={`${s.purpose}-${s.provider}`} title={s.last_error ?? undefined}>
                      <td>{s.purpose}</td>
                      <td>{s.provider}</td>
                      <td className="num">{s.calls}</td>
                      <td className={`num${s.errors ? " bad" : ""}`}>{s.errors || "–"}</td>
                      <td className="num">{s.tokens.toLocaleString()}</td>
                      <td className="num">{(s.avg_ms / 1000).toFixed(1)}s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {details.data.steps.some((s) => s.errors) && (
            <p className="muted small">Failed calls were retried or sent to another provider; hover a row to see its last error.</p>
          )}
          {details.data.traces.length > 0 && (
            <div className="traces">
              <p className="traces-head">What the AI was sent and answered</p>
              {details.data.traces.map((t) => <Trace key={t.id} trace={t} />)}
            </div>
          )}
        </>
      )}
    </details>
  );
}

function Trace({ trace }: { trace: LlmTrace }) {
  return (
    <details className="trace">
      <summary>
        <ChevronRight size={13} className="chev" />
        <strong>{trace.purpose}</strong>
        <span className="muted"> · {trace.provider} · {trace.tokens.toLocaleString()} tokens · {(trace.latency_ms / 1000).toFixed(1)}s · {trace.created_at.slice(11, 19)}</span>
      </summary>
      <p className="trace-label">Sent</p>
      <pre className="trace-text">{trace.prompt}</pre>
      <p className="trace-label">Answered</p>
      <pre className="trace-text">{trace.response ?? "(no answer text)"}</pre>
    </details>
  );
}
