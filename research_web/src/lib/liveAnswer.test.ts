import { describe, expect, it } from "vitest";
import type { AnswerPart } from "../api/types";
import { applyEvent, reveal, typedLength, type Live } from "./liveAnswer";

const start: Live = { question: "Did creatine help?", stage: null, sentences: [] };
const checked = (text: string, flags: string[] = []): AnswerPart => ({
  text, flags,
  citation: { paper_id: 10, rank: 4, label: "(Brose et al., 2003)", section: "results", quote: "greater", chunk_id: 1,
              passage: "CrM resulted in significantly greater increases" },
});

describe("applyEvent", () => {
  it("grows a sentence's text, then replaces it with the checked part", () => {
    let live = applyEvent(start, "stage", { step: "writing" });
    live = applyEvent(live, "text", { index: 0, text: "Brose" });
    live = applyEvent(live, "text", { index: 0, text: "Brose et al. found" });
    expect(live.sentences).toEqual([{ text: "Brose et al. found" }]);
    live = applyEvent(live, "sentence", { index: 0, part: checked("Brose et al. found gains.") });
    expect(live.sentences[0].part?.citation?.rank).toBe(4);
    live = applyEvent(live, "text", { index: 0, text: "late text" });        // a checked sentence no longer changes
    expect(live.sentences[0].text).toBe("Brose et al. found gains.");
    expect(live.stage).toEqual({ step: "writing" });
  });

  it("keeps the server's numbering when a repeated sentence is dropped, and forgets all on reset", () => {
    let live = applyEvent(start, "sentence", { index: 0, part: checked("One.") });
    live = applyEvent(live, "text", { index: 1, text: "Same point" });
    live = applyEvent(live, "sentence", { index: 1, part: null });
    live = applyEvent(live, "text", { index: 2, text: "Two" });
    expect(live.sentences.map((s) => s.dropped ?? s.text)).toEqual(["One.", true, "Two"]);
    expect(applyEvent(live, "reset", {}).sentences).toEqual([]);
    expect(applyEvent(live, "something-new", {})).toBe(live);              // unknown events are ignored
  });
});

describe("reveal", () => {
  const parts = [checked("One two."), checked("Three four.", ["quote not found in the passage"])];

  it("shows the whole answer when not typing", () => {
    expect(reveal(parts).map((r) => r.text)).toEqual(["One two.", "Three four."]);
  });

  it("types across sentences; a citation or flag shows only once its sentence is checked and fully typed", () => {
    const half = reveal(parts, { shown: 13, checked: [true, true] });
    expect(half.map((r) => r.text)).toEqual(["One two.", "Thre"]);
    expect(half[0].part.citation?.rank).toBe(4);                            // typed + checked: its number shows
    expect(half[1].part.citation).toBeNull();                               // still typing: no number yet,
    expect(half[1].part.flags).toEqual([]);                                 // and no ⚠ yet either
    const all = reveal(parts, { shown: typedLength(parts), checked: [true, true] });
    expect(all[1].part.flags).toEqual(["quote not found in the passage"]);  // a failed check is never hidden
  });

  it("never shows a number for a sentence the server hasn't checked, however far typing got", () => {
    const r = reveal([checked("Done.")], { shown: 99, checked: [false] });
    expect(r[0].text).toBe("Done.");
    expect(r[0].part.citation).toBeNull();
  });
});
