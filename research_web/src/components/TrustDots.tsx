import type { ReportPaper } from "../api/types";
import { TRUST_MAX, trustText } from "../lib/format";

/** ●●●○○ plus the label; the tooltip lists the reasons. */
export function TrustDots({ paper, showLabel = true }: { paper: ReportPaper; showLabel?: boolean }) {
  const level = Math.max(0, Math.min(TRUST_MAX, paper.trust.level));
  return (
    <span className="trust" title={[trustText(paper), ...paper.trust.reasons].join("\n")}>
      <span className="dots" aria-hidden="true">
        {Array.from({ length: TRUST_MAX }, (_, i) => <i key={i} className={i < level ? "on" : ""} />)}
      </span>
      {showLabel && <span className="muted">{trustText(paper)}</span>}
    </span>
  );
}
