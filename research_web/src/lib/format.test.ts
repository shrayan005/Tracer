import { describe, expect, it } from "vitest";
import type { ReportPaper } from "../api/types";
import {
  authorsShort, bareCitation, describeNotes, formatSeconds, groupWarnings, highlightParts, jobTitle, orderTimings,
  paperLink, parseQuality, relativeDate, sortPapers, surname, trustText,
} from "./format";

const paper = (rank: number, title: string, extra: Partial<ReportPaper["paper"]> = {}, level = 3,
               base?: number): ReportPaper => ({
  rank, paper_id: rank, relevance: 8, relevance_reason: "", study_type: "lab_experiment",
  trust: { level, label: ["", "very low", "low", "moderate", "high", "very high"][level], reasons: [] },
  base_trust: base === undefined ? null : { level: base, label: ["", "very low", "low", "moderate", "high"][base], reasons: [] },
  score: 0.5, score_parts: {}, summary: null, apa: "", in_text: "", bibtex_key: "", bibtex: "", analysis: null,
  paper: { source: "db", title, authors: [], pub_types: [], is_retracted: false, ...extra },
}) as ReportPaper;

describe("author names", () => {
  it("shortens lists the way citations do", () => {
    expect(authorsShort([])).toBe("—");
    expect(authorsShort(["Kazutoshi Takahashi"])).toBe("Takahashi");
    expect(authorsShort(["Andrea Brose", "Gianni Parise"])).toBe("Brose & Parise");
    expect(authorsShort(["Andrea Brose", "Gianni Parise", "M A Tarnopolsky"])).toBe("Brose et al.");
  });
  it("finds the surname in the different formats sources use", () => {
    expect(surname("Brose A")).toBe("Brose");             // Europe PMC: surname + initials
    expect(surname("Smith, J.")).toBe("Smith");           // "surname, initials"
    expect(surname("M. A. Tarnopolsky")).toBe("Tarnopolsky");
  });
});

describe("papers", () => {
  const papers = [paper(1, "Beta", { year: 2020 }, 3), paper(2, "alpha", { year: null }, 4), paper(3, "Gamma", { year: 2018 }, 3)];

  it("sorts by any column, missing values last, ties in rank order", () => {
    expect(sortPapers(papers, { key: "title", dir: "asc" }).map((p) => p.rank)).toEqual([2, 1, 3]);
    expect(sortPapers(papers, { key: "year", dir: "desc" }).map((p) => p.rank)).toEqual([1, 3, 2]);
    expect(sortPapers(papers, { key: "year", dir: "asc" }).map((p) => p.rank)).toEqual([3, 1, 2]);
    expect(sortPapers(papers, { key: "trust", dir: "desc" }).map((p) => p.rank)).toEqual([2, 1, 3]);
    expect(papers.map((p) => p.rank)).toEqual([1, 2, 3]);           // the original is untouched
  });

  it("shows when the quality check changed the trust", () => {
    expect(trustText(paper(1, "x", {}, 4, 3))).toBe("moderate → high");
    expect(trustText(paper(1, "x", {}, 3, 3))).toBe("moderate");
    expect(trustText(paper(1, "x", {}, 3))).toBe("moderate");
  });

  it("links to the DOI first", () => {
    expect(paperLink(paper(1, "x", { doi: "10.1/abc", landing_url: "https://site" }))).toBe("https://doi.org/10.1/abc");
    expect(paperLink(paper(1, "x", { landing_url: "https://site" }))).toBe("https://site");
    expect(paperLink(paper(1, "x"))).toBeNull();
  });
});


describe("small helpers (7.8)", () => {
  it("writes dates like a file list", () => {
    const now = new Date(2026, 10, 23, 12);
    expect(relativeDate(new Date(2026, 10, 23, 8).toISOString(), now)).toBe("Today");
    expect(relativeDate(new Date(2026, 10, 22, 23).toISOString(), now)).toBe("Yesterday");
    expect(relativeDate(new Date(2026, 2, 4).toISOString(), now)).toBe("Mar 4");
    expect(relativeDate(new Date(2025, 2, 4).toISOString(), now)).toBe("Mar 4, 2025");
  });

  it("finds the quote inside its passage despite spacing and case", () => {
    const passage = "Fourteen weeks of training.\n  CrM supplementation resulted in significantly greater increases in fat-free mass.";
    expect(highlightParts(passage, "crm supplementation resulted in significantly")).toEqual([
      "Fourteen weeks of training. ", "CrM supplementation resulted in significantly", " greater increases in fat-free mass."]);
    expect(highlightParts(passage, "not in the passage at all")).toBeNull();
  });

  it("groups search warnings by source and says what happened", () => {
    const { bySource, other } = groupWarnings([
      "europepmc / 'creatine older adults': timed out after 20s",
      "europepmc / 'dose response': timed out after 20s",
      "europepmc / 'muscle aging': skipped, overall search budget of 30s reached",
      "only 4 papers passed screening",
    ]);
    expect(bySource).toHaveLength(1);
    expect(bySource[0].source).toBe("Europe PMC");
    expect(bySource[0].notes.map((n) => n.query)).toEqual(["creatine older adults", "dose response", "muscle aging"]);
    expect(describeNotes(bySource[0].notes)).toBe("2 searches timed out, 1 skipped (time budget reached)");
    expect(other).toEqual(["only 4 papers passed screening"]);
  });

  it("orders timings like the pipeline and formats them", () => {
    expect(orderTimings({ analysis: 26.3, discovery: 31.7, selection: 0, screening: 3.8 }).map(([k]) => k))
      .toEqual(["discovery", "screening", "analysis"]);
    expect(formatSeconds(3.8)).toBe("4s");
    expect(formatSeconds(64.5)).toBe("1 min 5s");
    expect(formatSeconds(120)).toBe("2 min");
  });

  it("reads the quality score from its trust reason", () => {
    expect(parseQuality("quality 5/6.5: control group ✓, randomised ✓")).toEqual({ score: 5, max: 6.5 });
    expect(parseQuality("meta analysis")).toBeNull();
    expect(bareCitation("(Yao et al., 2026)")).toBe("Yao et al., 2026");
  });

  it("names every kind of run", () => {
    expect(jobTitle({ kind: "research", params: { topic: "creatine" } })).toBe("creatine");
    expect(jobTitle({ kind: "index", params: { topic_id: 7 } })).toBe("Prepare topic #7 for questions");
    expect(jobTitle({ kind: "analyze", params: { topic_id: 7 } })).toBe("Analyse topic #7");
  });
});
