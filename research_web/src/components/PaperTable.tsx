import { ChevronDown, ChevronUp, File, FileText } from "lucide-react";
import { useMemo, useState } from "react";
import type { ReportPaper } from "../api/types";
import { authorsShort, type Sort, type SortKey, sortPapers } from "../lib/format";
import { TrustDots } from "./TrustDots";

type Props = { papers: ReportPaper[]; selected: number | null; onSelect: (rank: number) => void };

const COLUMNS: { key: SortKey; label: string; className?: string }[] = [
  { key: "rank", label: "#", className: "col-rank" },
  { key: "title", label: "Name" },
  { key: "authors", label: "Authors", className: "hide-small" },
  { key: "venue", label: "Publication", className: "hide-medium" },
  { key: "year", label: "Year", className: "hide-small col-year" },
  { key: "trust", label: "Trust" },
];

/** The paper list: click a header to sort, click a row (or ↑/↓ + Enter) to open the paper. */
export function PaperTable({ papers, selected, onSelect }: Props) {
  const [sort, setSort] = useState<Sort>({ key: "rank", dir: "asc" });
  const rows = useMemo(() => sortPapers(papers, sort), [papers, sort]);

  const toggle = (key: SortKey) =>
    setSort((s) => (s.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" }
                                  : { key, dir: key === "trust" || key === "year" ? "desc" : "asc" }));

  const onKey = (event: React.KeyboardEvent, index: number) => {
    const move = event.key === "ArrowDown" ? 1 : event.key === "ArrowUp" ? -1 : 0;
    if (move) {
      event.preventDefault();
      const next = rows[index + move];
      if (next) {
        onSelect(next.rank);
        (event.currentTarget.parentElement?.children[index + move] as HTMLElement | undefined)?.focus();
      }
    } else if (event.key === "Enter") onSelect(rows[index].rank);
  };

  return (
    <table className="table papers">
      <colgroup>
        <col className="w-rank" /><col /><col className="w-authors hide-small" />
        <col className="w-venue hide-medium" /><col className="w-year hide-small" /><col className="w-trust" />
      </colgroup>
      <thead>
        <tr>
          {COLUMNS.map((c) => (
            <th key={c.key} className={c.className} aria-sort={sort.key === c.key ? (sort.dir === "asc" ? "ascending" : "descending") : undefined}>
              <button className="th-button" onClick={() => toggle(c.key)}>
                {c.label}
                {sort.key === c.key && (sort.dir === "asc" ? <ChevronUp size={13} /> : <ChevronDown size={13} />)}
              </button>
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((p, index) => (
          <tr key={p.rank} tabIndex={0} className={p.rank === selected ? "selected" : undefined}
              onClick={() => onSelect(p.rank)} onKeyDown={(e) => onKey(e, index)}>
            <td className="col-rank muted">{p.rank}</td>
            <td>
              <div className="name" title={p.paper.title}>
                {p.summary?.based_on === "full text"
                  ? <FileText size={16} aria-label="full text" /> : <File size={16} className="muted" aria-label="abstract only" />}
                <span>{p.paper.title}</span>
              </div>
            </td>
            <td className="hide-small nowrap">{authorsShort(p.paper.authors ?? [])}</td>
            <td className="hide-medium cell-clip muted">{p.paper.venue ?? "—"}</td>
            <td className="hide-small col-year muted">{p.paper.year ?? "—"}</td>
            <td className="nowrap"><TrustDots paper={p} showLabel={false} /></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
