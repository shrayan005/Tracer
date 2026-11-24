// The answer while it streams (SSE step 5): pure functions, so they are tested without a browser.
import type { AnswerPart, AskResult, AskStage } from "../api/types";

/** One sentence as the server sends it: its text so far, then (once finished and checked) its part. */
export type LiveSentence = { text: string; part?: AnswerPart; dropped?: boolean };
export type Live = { question: string; stage: AskStage | null; sentences: LiveSentence[]; result?: AskResult };

/** Fold one server event into the live answer. Sentences keep the server's index (a dropped one stays as a gap). */
export function applyEvent(live: Live, event: string, data: any): Live {
  if (event === "stage") return { ...live, stage: data as AskStage };
  if (event === "reset") return { ...live, sentences: [] };              // invalid JSON: the server asks again
  if (event !== "text" && event !== "sentence") return live;            // unknown events: ignore (forward-compatible)
  const sentences = [...live.sentences];
  while (sentences.length <= data.index) sentences.push({ text: "" });
  if (event === "text" && !sentences[data.index].part) sentences[data.index] = { text: data.text };
  if (event === "sentence") {
    sentences[data.index] = data.part ? { text: data.part.text, part: data.part } : { text: "", dropped: true };
  }
  return { ...live, sentences };
}

/** `shown`: characters typed out so far; `checked`: per part, has the server checked it yet. */
export type Typing = { shown: number; checked: boolean[] };

/** Revealed text per part. A citation number or ⚠ appears only when its part is checked AND fully typed:
 *  the text may run ahead of the check, the proof never does. */
export function reveal(parts: AnswerPart[], typing?: Typing): { part: AnswerPart; text: string }[] {
  if (!typing) return parts.map((part) => ({ part, text: part.text }));
  let start = 0;
  return parts.flatMap((p, i) => {
    const text = p.text.slice(0, Math.max(0, typing.shown - start));
    start += p.text.length + 1;                                          // +1: the space between sentences
    if (!text) return [];
    const ready = typing.checked[i] && text.length === p.text.length;
    return [{ part: ready ? p : { ...p, citation: null, flags: [] }, text }];
  });
}

/** Characters needed to type out these parts (as counted by reveal). */
export const typedLength = (parts: AnswerPart[]) => parts.reduce((n, p) => n + p.text.length + 1, 0);
