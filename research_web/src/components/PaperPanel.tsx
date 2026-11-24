import {
  AlertTriangle, ArrowRight, Check, ExternalLink, FileText, Link as LinkIcon, Minus, TrendingDown, TrendingUp, X,
  type LucideIcon,
} from "lucide-react";
import type { Citation, Claim, ReportPaper, ResearchReport } from "../api/types";
import {
  authorsShort, bareCitation, capitalize, FACT_NAMES, highlightParts, paperLink, parseQuality, STANCE_TEXT, studyName,
  trustText,
} from "../lib/format";
import { CopyButton } from "./CopyButton";
import { TrustDots } from "./TrustDots";

type Props = { report: ResearchReport; paper: ReportPaper; onClose: () => void; onOpen: (rank: number) => void;
               highlight?: Citation | null };

const DIRECTION: Record<Claim["direction"], { icon: LucideIcon; text: string; tone: string }> = {
  increase: { icon: TrendingUp, text: "increased", tone: "up" },
  decrease: { icon: TrendingDown, text: "decreased", tone: "down" },
  "no difference": { icon: Minus, text: "did not change", tone: "flat" },
  association: { icon: LinkIcon, text: "was associated", tone: "flat" },
  other: { icon: ArrowRight, text: "changed", tone: "flat" },
};

/** Everything about one paper, from the topic already loaded (no extra request). */
export function PaperPanel({ report, paper: p, onClose, onOpen, highlight }: Props) {
  const link = paperLink(p);
  const s = p.summary?.summary ?? null;
  const basedOn = p.summary?.based_on ?? "title only";
  const flags = [...(p.summary?.flags ?? []), ...(p.analysis?.flags ?? [])];
  const appraisal = p.analysis?.analysis.appraisal;
  const claims = p.analysis?.analysis.claims ?? [];
  const connections = (report.connections ?? []).filter((c) => c.members.some((m) => m.paper === p.paper_id));
  const rankOf = new Map(report.papers.map((x) => [x.paper_id, x]));
  const quality = p.trust.reasons.map(parseQuality).find(Boolean) ?? null;
  // the quality line lists the checklist again: with an appraisal below, show it as a score only
  const reasons = p.trust.reasons.filter((r) => !(appraisal && parseQuality(r)));

  return (
    <article className="paper-panel">
      <div className="panel-head">
        <div className="pills">
          <span className="pill">#{p.rank}</span>
          <span className="pill pill-brand">{studyName(p.study_type)}</span>
          <span className="pill">{basedOn === "full text" ? <><FileText size={12} /> Full text</> : basedOn === "abstract" ? "Abstract only" : "Title only"}</span>
        </div>
        <button className="icon-button" onClick={onClose} aria-label="Close"><X size={16} /></button>
      </div>
      <h2>{p.paper.title}</h2>
      <p className="muted meta">
        {authorsShort(p.paper.authors ?? [])} · {p.paper.venue ?? "—"} · {p.paper.year ?? "—"}
        {p.paper.is_retracted && <strong className="danger"> · RETRACTED</strong>}
      </p>
      <div className="chips">
        {link && <a className="chip" href={link} target="_blank" rel="noreferrer"><ExternalLink size={13} /> Paper</a>}
        {p.paper.pdf_url && <a className="chip" href={p.paper.pdf_url} target="_blank" rel="noreferrer"><ExternalLink size={13} /> PDF</a>}
        <CopyButton text={p.apa} label="APA" />
        <CopyButton text={p.bibtex} label="BibTeX" />
      </div>

      {highlight && <QuotedPassage citation={highlight} />}

      <section className="trust-card">
        <div className="trust-grid">
          <div>
            <span className="mini-label">Trust</span>
            <div className="trust-level"><TrustDots paper={p} showLabel={false} /><strong>{capitalize(trustText(p))}</strong></div>
          </div>
          <div>
            <span className="mini-label">Relevance</span>
            <div className="trust-level"><Meter value={p.relevance} max={10} /><strong>{p.relevance}/10</strong></div>
          </div>
        </div>
        {quality && (
          <div className="quality-row">
            <span>Study quality</span>
            <Meter value={quality.score} max={quality.max} wide />
            <span className="muted">{quality.score} / {quality.max}</span>
          </div>
        )}
        <ul className="reason-list">
          {reasons.map((r, i) => <li key={i}>{capitalize(r)}</li>)}
        </ul>
        <p className="muted small">{p.relevance_reason}</p>
      </section>

      {flags.length > 0 && (
        <section className="flags">
          {flags.map((f, i) => <p key={i}><AlertTriangle size={13} /> {f}</p>)}
        </section>
      )}

      <section>
        <h3>Summary <span className="h3-note">from the {basedOn}</span></h3>
        {!s && <p className="muted">No summary: there was no abstract or free full text.</p>}
        {s && (
          <>
            <p className="tldr">{s.tldr}</p>
            <div className="field-block"><span className="mini-label">Objective</span><p>{s.objective}</p></div>
            <div className="field-block"><span className="mini-label">Methods</span><p>{s.methods}</p></div>
            <span className="mini-label">Key findings</span>
            <ol className="findings">
              {s.key_findings.map((f, i) => (
                <li key={i}>
                  <span className="finding-n">{i + 1}</span>
                  <div>{f.statement}{f.quote && <blockquote>“{f.quote}”</blockquote>}</div>
                </li>
              ))}
            </ol>
            {(s.limitations ?? []).length > 0 && (
              <>
                <span className="mini-label">Limitations the authors state</span>
                <ul className="bullets">{(s.limitations ?? []).map((l, i) => <li key={i}>{l}</li>)}</ul>
              </>
            )}
            <div className="field-block why"><span className="mini-label">Why it's here</span><p>{s.why_relevant}</p></div>
          </>
        )}
      </section>

      {appraisal && (
        <section>
          <h3>How the study was done</h3>
          <div className="study-facts">
            <div><span className="mini-label">Participants</span><strong>{appraisal.sample_size ?? "—"}</strong></div>
            <div><span className="mini-label">Sex</span><strong>{capitalize(appraisal.sex)}</strong></div>
            <div><span className="mini-label">Duration</span><strong>{capitalize(appraisal.duration)}</strong></div>
          </div>
          <ul className="checklist">
            {Object.entries(FACT_NAMES).map(([key, name]) => {
              const fact = appraisal[key as keyof typeof appraisal] as { value: string; quote?: string | null };
              const Icon = fact.value === "yes" ? Check : fact.value === "no" ? X : Minus;
              return (
                <li key={key} title={fact.quote ?? fact.value}>
                  <span className={`check check-${fact.value.replace(" ", "-")}`}><Icon size={11} strokeWidth={3} /></span>
                  <span>{name}{fact.value !== "yes" && fact.value !== "no" && <span className="muted"> · {fact.value}</span>}</span>
                </li>
              );
            })}
          </ul>
        </section>
      )}

      {claims.length > 0 && (
        <section>
          <h3>Claims <span className="h3-note">{claims.length}</span></h3>
          <ul className="claim-list">
            {claims.map((c, i) => {
              const d = DIRECTION[c.direction] ?? DIRECTION.other;
              return (
                <li key={i}>
                  <span className={`dir dir-${d.tone}`}><d.icon size={13} /></span>
                  <div>
                    <strong>{capitalize(c.outcome)}</strong> {d.text} with {c.factor}
                    <div className="muted small">{c.population}{c.magnitude ? ` · ${c.magnitude}` : ""}</div>
                  </div>
                </li>
              );
            })}
          </ul>
        </section>
      )}

      {connections.length > 0 && (
        <section>
          <h3>Connections</h3>
          {connections.map((c, i) => (
            <div key={i} className="mini-connection">
              <span className={`relation relation-${c.relation}`}>{c.relation}</span>
              <p>{c.question}</p>
              <div className="paper-tags">
                {otherPapers(c.members, p.paper_id).map((m) => {
                  const other = rankOf.get(m.paper);
                  return other ? (
                    <button key={m.paper} className="tag tag-button" onClick={() => onOpen(other.rank)}>
                      {bareCitation(other.in_text)} · {m.stances.map((s) => STANCE_TEXT[s] ?? s).join(" / ")}
                    </button>
                  ) : null;
                })}
              </div>
            </div>
          ))}
        </section>
      )}
    </article>
  );
}

/** The OTHER papers of a connection, once each (a paper can join with several claims), with their stances. */
function otherPapers(members: { paper: number; stance: string }[], self: number) {
  const byPaper = new Map<number, string[]>();
  for (const m of members) {
    if (m.paper === self) continue;
    const stances = byPaper.get(m.paper) ?? [];
    if (!stances.includes(m.stance)) stances.push(m.stance);
    byPaper.set(m.paper, stances);
  }
  return [...byPaper].map(([paper, stances]) => ({ paper, stances }));
}

function Meter({ value, max, wide = false }: { value: number; max: number; wide?: boolean }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  return <span className={wide ? "meter meter-wide" : "meter"}><i style={{ width: `${pct}%` }} /></span>;
}

/** The passage an answer cited, with the quoted words highlighted (or the quote alone if it isn't found word
 *  for word: then the answer carries a ⚠ too). */
function QuotedPassage({ citation }: { citation: Citation }) {
  const parts = citation.passage ? highlightParts(citation.passage, citation.quote) : null;
  return (
    <section className="quoted">
      <h3>Cited passage <span className="h3-note">{citation.section}</span></h3>
      {parts ? (
        <p className="passage">{parts[0]}<mark>{parts[1]}</mark>{parts[2]}</p>
      ) : (
        <p className="passage"><mark>“{citation.quote}”</mark></p>
      )}
    </section>
  );
}
