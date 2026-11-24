// Small pure helpers for showing a report: names, labels, links, sorting. No React: unit-tested (format.test.ts).
import type { ReportPaper } from "../api/types";

/** "Takahashi et al." from ["Kazutoshi Takahashi", "Shinya Yamanaka", ...] (or "Brose A", "Smith, J."). */
export function authorsShort(authors: string[]): string {
  if (!authors.length) return "—";
  const first = surname(authors[0]);
  if (authors.length === 1) return first;
  if (authors.length === 2) return `${first} & ${surname(authors[1])}`;
  return `${first} et al.`;
}

export function surname(name: string): string {
  const clean = name.trim();
  if (clean.includes(",")) return clean.split(",")[0].trim();            // "Smith, J."
  const parts = clean.split(/\s+/);
  const last = parts[parts.length - 1];
  // "Brose A" / "Brose AB": the initials come last, so the surname is the first word
  if (parts.length > 1 && /^[A-Z]{1,3}\.?$/.test(last.replace(/\./g, ""))) return parts[0];
  return last;
}

export const STUDY_NAMES: Record<string, string> = {
  meta_analysis: "Meta-analysis", systematic_review: "Systematic review", narrative_review: "Narrative review",
  randomized_trial: "Randomized trial", cohort: "Cohort", case_control: "Case-control",
  cross_sectional: "Cross-sectional", case_report: "Case report", lab_experiment: "Lab experiment",
  computational: "Computational", theoretical: "Theoretical", other: "Other",
};
export const studyName = (type: string) => STUDY_NAMES[type] ?? type.replace(/_/g, " ");

export const TRUST_MAX = 5;

/** "moderate", or "moderate → high" when the quality check (full text) changed it. */
export function trustText(p: ReportPaper): string {
  const base = p.base_trust;
  if (!base || base.level === p.trust.level) return p.trust.label;
  return `${base.label} → ${p.trust.label}`;
}

/** The best link to a paper: its DOI, else the page a source gave. */
export function paperLink(p: ReportPaper): string | null {
  if (p.paper.doi) return `https://doi.org/${p.paper.doi}`;
  return p.paper.landing_url ?? null;
}

export type SortKey = "rank" | "title" | "authors" | "venue" | "year" | "trust";
export type Sort = { key: SortKey; dir: "asc" | "desc" };

/** A sorted COPY; missing values always last; ties keep the report's rank order. */
export function sortPapers(papers: ReportPaper[], sort: Sort): ReportPaper[] {
  const value = (p: ReportPaper): string | number | null => {
    switch (sort.key) {
      case "rank": return p.rank;
      case "title": return p.paper.title.toLowerCase();
      case "authors": return p.paper.authors?.length ? surname(p.paper.authors[0]).toLowerCase() : null;
      case "venue": return p.paper.venue?.toLowerCase() ?? null;
      case "year": return p.paper.year ?? null;
      case "trust": return p.trust.level;
    }
  };
  const sign = sort.dir === "asc" ? 1 : -1;
  return [...papers].sort((a, b) => {
    const va = value(a), vb = value(b);
    if (va === vb) return a.rank - b.rank;
    if (va === null) return 1;
    if (vb === null) return -1;
    return (va < vb ? -1 : 1) * sign;
  });
}

export const RELATION_TEXT: Record<string, string> = { agree: "agree", contradict: "contradict", mixed: "mixed" };
export const STANCE_TEXT: Record<string, string> = { supports: "yes", opposes: "no", "no effect": "no effect" };

export const FACT_NAMES: Record<string, string> = {
  control_group: "Control group", randomized: "Randomised", blinded: "Blinded",
  statistics_reported: "Statistics reported", effect_sizes: "Effect sizes",
  conflicts_declared: "Conflicts declared", data_available: "Data available",
};

/** What a run is about, for lists and headings. */
export function jobTitle(job: { kind: string; params: Record<string, unknown> }): string {
  if (job.kind === "research") return String(job.params.topic ?? "Research");
  if (job.kind === "index") return `Prepare topic #${job.params.topic_id} for questions`;
  return `Analyse topic #${job.params.topic_id}`;
}

/** "Today", "Yesterday", "Mar 4", or "Mar 4, 2025" for another year (like a file list). */
export function relativeDate(iso: string, now: Date = new Date()): string {
  const date = new Date(iso);
  const day = (d: Date) => Date.UTC(d.getFullYear(), d.getMonth(), d.getDate());
  const days = Math.round((day(now) - day(date)) / 86_400_000);
  if (days === 0) return "Today";
  if (days === 1) return "Yesterday";
  const sameYear = date.getFullYear() === now.getFullYear();
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", ...(sameYear ? {} : { year: "numeric" }) });
}

export const SOURCE_NAMES: Record<string, string> = {
  openalex: "OpenAlex", europepmc: "Europe PMC", pubmed: "PubMed", arxiv: "arXiv",
  semantic_scholar: "Semantic Scholar", unpaywall: "Unpaywall",
};
export const sourceName = (s: string) => SOURCE_NAMES[s] ?? s;

/** "Sports science" from "sports science". */
export const capitalize = (s: string) => (s ? s[0].toUpperCase() + s.slice(1) : s);

/** "(Yao et al., 2026)" -> "Yao et al., 2026" */
export const bareCitation = (s: string) => s.replace(/^\(|\)$/g, "");

export type SearchNote = { source: string; query: string; reason: string };
export type WarningGroups = { bySource: { source: string; notes: SearchNote[] }[]; other: string[] };

/** The search's warnings ("europepmc / 'creatine aging': timed out after 20s") grouped by source, so the
 *  page can say "Europe PMC: 3 searches timed out" instead of listing log lines. Anything else stays as is. */
export function groupWarnings(warnings: string[]): WarningGroups {
  const groups = new Map<string, SearchNote[]>();
  const other: string[] = [];
  for (const w of warnings) {
    const m = /^([\w-]+) \/ '(.*)': (.+)$/.exec(w);
    if (!m) { other.push(w); continue; }
    const source = sourceName(m[1]);
    groups.set(source, [...(groups.get(source) ?? []), { source, query: m[2], reason: m[3] }]);
  }
  return { bySource: [...groups].map(([source, notes]) => ({ source, notes })), other };
}

/** "3 searches timed out, 1 skipped" for one source's notes. */
export function describeNotes(notes: SearchNote[]): string {
  const timedOut = notes.filter((n) => n.reason.startsWith("timed out")).length;
  const skipped = notes.filter((n) => n.reason.startsWith("skipped")).length;
  const failed = notes.length - timedOut - skipped;
  const parts = [
    timedOut && `${timedOut} ${timedOut === 1 ? "search" : "searches"} timed out`,
    skipped && `${skipped} skipped (time budget reached)`,
    failed && `${failed} failed`,
  ].filter(Boolean);
  return parts.join(", ");
}

const STAGE_ORDER = ["planning", "plan", "discovery", "screening", "selection", "full text", "fulltext",
                     "summaries", "analysis", "index"];

/** The pipeline's timings in pipeline order (unknown stages last), without zero-second stages. */
export function orderTimings(timings: Record<string, number>): [string, number][] {
  const rank = (k: string) => (STAGE_ORDER.includes(k) ? STAGE_ORDER.indexOf(k) : STAGE_ORDER.length);
  return Object.entries(timings).filter(([, v]) => v > 0).sort((a, b) => rank(a[0]) - rank(b[0]));
}

/** "45s", "1 min 5s" */
export function formatSeconds(s: number): string {
  const total = Math.round(s);
  if (total < 60) return `${total}s`;
  const rest = total % 60;
  return `${Math.floor(total / 60)} min${rest ? ` ${rest}s` : ""}`;
}

/** The trust reason "quality 5/6.5: control group ✓, ..." as numbers, or null for any other reason. */
export function parseQuality(reason: string): { score: number; max: number } | null {
  const m = /^quality ([\d.]+)\/([\d.]+)/.exec(reason);
  return m ? { score: Number(m[1]), max: Number(m[2]) } : null;
}

/** Split a passage around its quote so the quote can be highlighted: [before, quote, after], or null when the
 *  quote isn't found word for word (spacing and letter case are ignored). */
export function highlightParts(passage: string, quote: string): [string, string, string] | null {
  const text = passage.replace(/\s+/g, " ").trim();
  const needle = quote.replace(/\s+/g, " ").trim().replace(/^[“"']|[”"'.]$/g, "");
  if (needle.length < 3) return null;
  const at = text.toLowerCase().indexOf(needle.toLowerCase());
  if (at < 0) return null;
  return [text.slice(0, at), text.slice(at, at + needle.length), text.slice(at + needle.length)];
}
