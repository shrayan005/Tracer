// Friendly names for the types GENERATED from the backend (schema.d.ts, `npm run gen:api`).
// If the backend changes a field, `npm run typecheck` shows every place in the UI that must change.
import type { components } from "./schema";

type S = components["schemas"];
export type UserInfo = S["UserInfo"];
export type TopicSummary = S["TopicSummary"];
export type JobInfo = S["JobInfo"];
export type JobDetails = S["JobDetails"];
export type LlmTrace = S["LlmTrace"];
export type UsageInfo = S["UsageInfo"];
export type ResearchReport = S["ResearchReport"];
export type ReportPaper = S["ReportPaper"];
export type PaperDetail = S["PaperDetail"];
export type Answer = S["Answer"];
export type AnswerPart = S["AnswerPart"];
export type Citation = S["Citation"];
export type AskResult = S["AskResult"];
export type ChatInfo = S["ChatInfo"];
export type ChatMessageInfo = S["ChatMessageInfo"];
export type Readiness = S["Readiness"];
export type SearchHit = S["SearchHit"];
export type Connection = S["Connection"];
export type Claim = S["Claim"];

/** A step of a streamed answer (POST /api/topics/{id}/ask/stream sends these as `stage` events). Written by hand:
 *  OpenAPI can't describe the events inside a stream. Mirrors AskService._ask_steps in the backend. */
export type AskStage =
  | { step: "searching"; papers: number }
  | { step: "reading"; passages: number; papers: number }
  | { step: "writing" }
  | { step: "checking" };
