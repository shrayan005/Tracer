import {
  CheckCheck, ChevronDown, ChevronRight, CircleSlash, Clock, Download, FlaskConical, GitCompare, Info, Lightbulb, Minus, Search,
  Check, X, type LucideIcon,
} from "lucide-react";
import type { ReactNode } from "react";
import type { Connection, ResearchReport } from "../api/types";
import {
  bareCitation, capitalize, describeNotes, formatSeconds, groupWarnings, orderTimings, relativeDate, sourceName,
} from "../lib/format";

type Open = { onOpen: (rank: number) => void };

/** The title block every report view starts with. */
function ViewHeader({ eyebrow, title, sub, actions }: { eyebrow: string; title: string; sub?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="doc-head">
      <div>
        <p className="doc-eyebrow">{eyebrow}</p>
        <h2 className="doc-title">{title}</h2>
        {sub && <p className="doc-sub">{sub}</p>}
      </div>
      {actions && <div className="doc-actions">{actions}</div>}
    </header>
  );
}

function Label({ children, aside }: { children: ReactNode; aside?: ReactNode }) {
  return <div className="doc-label"><span>{children}</span>{aside && <span className="doc-label-aside">{aside}</span>}</div>;
}

/* ---------------------------------------------------------------- connections */

const RELATION: Record<Connection["relation"], { label: string; icon: LucideIcon }> = {
  agree: { label: "Papers agree", icon: CheckCheck },
  contradict: { label: "Papers disagree", icon: X },
  mixed: { label: "Mixed evidence", icon: GitCompare },
};
const STANCE: Record<string, { label: string; icon: LucideIcon; tone: string }> = {
  supports: { label: "Yes", icon: Check, tone: "yes" },
  opposes: { label: "No", icon: X, tone: "no" },
  "no effect": { label: "No effect", icon: Minus, tone: "none" },
};

/** How the papers answer the same questions: agree / contradict / mixed (computed by code, step 5.3). */
export function ConnectionsView({ report, onOpen }: { report: ResearchReport } & Open) {
  const connections = report.connections ?? [];
  if (!connections.length) {
    return <div className="empty"><strong>No connections</strong>
      {report.gaps ? "These papers don't answer the same questions." : "Run “Re-analyse” to find them."}</div>;
  }
  return (
    <div className="doc">
      <ViewHeader eyebrow="Connections" title="Where the papers meet"
                  sub="Questions that several papers answer, and how their results line up. Matched by code from each paper's claims." />
      {connections.map((c, i) => <ConnectionCard key={i} c={c} report={report} onOpen={onOpen} />)}
    </div>
  );
}

function ConnectionCard({ c, report, onOpen }: { c: Connection; report: ResearchReport } & Open) {
  const byId = new Map(report.papers.map((p) => [p.paper_id, p]));
  const rel = RELATION[c.relation];
  const counts = ["supports", "opposes", "no effect"]
    .map((s) => ({ s, n: c.members.filter((m) => m.stance === s).length }))
    .filter((x) => x.n > 0);
  const papers = new Set(c.members.map((m) => m.paper)).size;
  return (
    <section className="doc-card conn">
      <div className="conn-top">
        <span className={`relation relation-${c.relation}`}><rel.icon size={13} /> {rel.label}</span>
        <span className="muted small">{papers} papers · {c.members.length} results</span>
      </div>
      <h3 className="conn-q">{c.question}</h3>
      <p className="conn-finding">{c.finding}</p>

      <div className="tally" aria-label={counts.map((x) => `${x.n} ${STANCE[x.s].label}`).join(", ")}>
        <div className="tally-bar">
          {counts.map((x) => <i key={x.s} className={`t-${STANCE[x.s].tone}`} style={{ flexGrow: x.n }} />)}
        </div>
        <div className="tally-legend">
          {counts.map((x) => <span key={x.s}><i className={`t-${STANCE[x.s].tone}`} />{x.n} {STANCE[x.s].label.toLowerCase()}</span>)}
        </div>
      </div>

      <ul className="conn-rows">
        {c.members.map((m, i) => {
          const p = byId.get(m.paper);
          const claim = m.claim_snapshot;
          const st = STANCE[m.stance] ?? { label: m.stance, icon: Minus, tone: "none" };
          return (
            <li key={i} className="conn-row">
              <span className={`stance-pill s-${st.tone}`}><st.icon size={12} strokeWidth={2.5} />{st.label}</span>
              <div className="conn-body">
                {p ? <button className="paper-link" onClick={() => onOpen(p.rank)}>{bareCitation(p.in_text)}</button>
                   : <span>a paper</span>}
                {claim && <span className="conn-detail">{capitalize(claim.outcome)} · {claim.population}</span>}
              </div>
              {claim?.magnitude && <span className="conn-mag" title={claim.magnitude}>{claim.magnitude}</span>}
            </li>
          );
        })}
      </ul>

      {c.why_differ && (
        <div className="callout">
          <Info size={15} />
          <div><strong>Why they differ</strong><p>{c.why_differ}</p></div>
        </div>
      )}
    </section>
  );
}

/* ---------------------------------------------------------------- gaps */

/** What the evidence is missing (counted by code, step 5.4). */
export function GapsView({ report }: { report: ResearchReport }) {
  const g = report.gaps;
  if (!g) return <div className="empty"><strong>Not analysed yet</strong>Run “Re-analyse” to find the gaps.</div>;
  const next = g.write_up?.next_studies ?? [];
  const shared = g.shared_limitations ?? [];
  return (
    <div className="doc">
      <ViewHeader eyebrow="Evidence gaps" title="What the evidence leaves open"
                  sub={`Across ${g.n_papers} analysed papers: what is missing, what the authors admit, and what to study next.`} />

      {g.write_up && (
        <section className="doc-lead">
          <Label>In short</Label>
          <p>{g.write_up.summary}</p>
        </section>
      )}

      <section className="doc-card">
        <Label aside="counted from the papers">What's missing</Label>
        {g.gaps.length ? (
          <ul className="icon-rows">
            {g.gaps.map((x, i) => <li key={i}><span className="icon-sq amber"><CircleSlash size={14} /></span>{x}</li>)}
          </ul>
        ) : <p className="muted">Nothing obvious: the papers report the basics.</p>}
      </section>

      {shared.length > 0 && (
        <section className="doc-card">
          <Label aside={`${g.papers_with_limitations} papers state limitations`}>Limitations several authors admit</Label>
          <ul className="icon-rows">
            {shared.map((l, i) => (
              <li key={i}>
                <span className="icon-sq"><FlaskConical size={14} /></span>
                <div><strong>{capitalize(l.theme)}</strong>
                  <div className="paper-tags">{l.papers.map((p) => <span key={p} className="tag">{bareCitation(p)}</span>)}</div></div>
              </li>
            ))}
          </ul>
        </section>
      )}

      {next.length > 0 && (
        <section>
          <Label>Studies that would help</Label>
          <div className="idea-grid">
            {next.map((s, i) => (
              <div key={i} className="idea">
                <span className="icon-sq brand"><Lightbulb size={14} /></span>
                <p>{s}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {(g.notes ?? []).map((n, i) => <div key={i} className="notice notice-info"><Info size={15} /><span>{n}</span></div>)}
    </div>
  );
}

/* ---------------------------------------------------------------- overview */

const FUNNEL: Record<string, [string, string]> = {
  found: ["Found", "in the databases"],
  prefiltered: ["Prefiltered", "by keywords"],
  screened: ["Screened", "abstracts read"],
  shortlisted: ["Shortlisted", "full text tried"],
  selected: ["Selected", "full text first"],
  full_text: ["Full text", "read in full"],
};

/** The run behind the report: brief, funnel, queries, time and search notes, plus the report to download. */
export function OverviewView({ report }: { report: ResearchReport }) {
  const counts = report.counts as Record<string, number>;
  const funnel = Object.keys(FUNNEL).filter((k) => k in counts);
  const timings = orderTimings(report.timings_s as Record<string, number>);
  const total = timings.reduce((s, [, v]) => s + v, 0);
  const { bySource, other } = groupWarnings(report.warnings);
  const notes = [...other, ...(report.synthesis_flags ?? [])];
  // rounds where the search agent actually searched (a "done" round adds nothing to show)
  const agentRounds = (report.search_agent ?? []).filter((r) => (r.new_queries ?? []).length > 0);
  const meta = [
    report.created_at && relativeDate(report.created_at),
    `${report.papers.length} papers`,
    report.sources_used.map(sourceName).join(", "),
  ].filter(Boolean).join(" · ");

  return (
    <div className="doc">
      <ViewHeader eyebrow={capitalize(report.plan.field)} title={capitalize(report.topic)} sub={meta}
                  actions={<a className="btn-outline" href={`/api/topics/${report.topic_id}/report.md`} download>
                    <Download size={14} /> Report (.md)</a>} />

      <section className="doc-lead">
        <Label>Research brief</Label>
        <p>{report.plan.research_brief}</p>
      </section>

      <section>
        <Label>From search to selection</Label>
        <div className="funnel-row">
          {funnel.map((k, i) => (
            <div key={k} className="funnel-step">
              <div className="stat">
                <strong>{counts[k]}</strong>
                <span>{FUNNEL[k][0]}</span>
                <small>{FUNNEL[k][1]}</small>
              </div>
              {i < funnel.length - 1 && <ChevronRight size={16} className="funnel-arrow" />}
            </div>
          ))}
        </div>
      </section>

      <section className="doc-card">
        <Label aside={`${report.plan.queries.length + agentRounds.reduce((n, r) => n + (r.new_queries?.length ?? 0), 0)} queries`}>Searches</Label>
        <div className="query-list">
          {report.plan.queries.map((q) => <span key={q} className="query"><Search size={13} />{q}</span>)}
        </div>
        {agentRounds.map((r) => (
          <div key={r.round} className="agent-round">
            <p className="agent-round-head">
              <strong>Search agent · round {r.round}</strong>
              <span className="muted"> +{r.new_papers ?? 0} papers</span>
            </p>
            <p className="agent-round-gap muted">Gap it saw: {r.gaps}</p>
            <div className="query-list">
              {(r.new_queries ?? []).map((q) => <span key={q} className="query query-agent"><Search size={13} />{q}</span>)}
            </div>
          </div>
        ))}
      </section>

      {total > 0 && (
        <section className="doc-card">
          <Label aside={<><Clock size={13} /> {formatSeconds(total)} in total</>}>Time</Label>
          <div className="time-bar">
            {timings.map(([k, v], i) => <i key={k} className={`seg-${i % 6}`} style={{ flexGrow: v }} title={`${k}: ${formatSeconds(v)}`} />)}
          </div>
          <ul className="time-legend">
            {timings.map(([k, v], i) => (
              <li key={k}><i className={`seg-${i % 6}`} /><span>{capitalize(k)}</span><span className="muted">{formatSeconds(v)}</span></li>
            ))}
          </ul>
        </section>
      )}

      {(bySource.length > 0 || notes.length > 0) && (
        <section className="doc-card">
          <Label>Search notes</Label>
          <ul className="icon-rows">
            {bySource.map(({ source, notes: n }) => (
              <li key={source}>
                <span className="icon-sq"><Clock size={14} /></span>
                <details className="note-details">
                  <summary><strong>{source}</strong><span className="muted">{describeNotes(n)}</span><ChevronDown size={14} /></summary>
                  <ul>{n.map((x) => <li key={x.query}>“{x.query}” <span className="muted">· {x.reason}</span></li>)}</ul>
                </details>
              </li>
            ))}
            {notes.map((w, i) => <li key={i}><span className="icon-sq amber"><Info size={14} /></span>{capitalize(w)}</li>)}
          </ul>
          {bySource.length > 0 && (
            <p className="muted small doc-foot">Slow sources are skipped so a topic never waits on one database; the other sources still ran.</p>
          )}
        </section>
      )}
    </div>
  );
}
