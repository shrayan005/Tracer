import { AlertTriangle, ArrowUp, Check, Copy, Loader2, Plus, Sparkle, Trash2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { ApiError } from "../api/client";
import { isActive, useAsk, useChat, useChats, useDeleteChat, useJobs, usePrepare, useReadiness, useTopic } from "../api/queries";
import type { Answer, AnswerPart, AskStage, ChatMessageInfo, Citation } from "../api/types";
import { applyEvent, reveal, typedLength, type Live, type Typing } from "../lib/liveAnswer";
import { useToast } from "./Toast";

export type OpenPaper = (rank: number, citation?: Citation) => void;
type Props = { topicId: number; onOpen: OpenPaper };

/** Ask this topic's papers. Answers come only from them; every sentence cites a passage whose quote was checked. */
export function AskView({ topicId, onOpen }: Props) {
  const readiness = useReadiness(topicId);
  if (readiness.isPending) return <div className="ask"><div className="thread"><div className="thread-inner"><Loader2 size={16} className="spin muted" /></div></div></div>;
  if (readiness.error) return <div className="empty"><strong>Can't ask here</strong>{readiness.error.message}</div>;
  if (!readiness.data.prepared) return <Prepare topicId={topicId} />;
  return <Conversation topicId={topicId} onOpen={onOpen} prepared={readiness.data.prepared} papers={readiness.data.papers} />;
}

function Prepare({ topicId }: { topicId: number }) {
  const prepare = usePrepare(topicId);
  const jobs = useJobs();
  const running = (jobs.data ?? []).find((j) => j.kind === "index" && j.params.topic_id === topicId && isActive(j));
  return (
    <div className="empty empty-card">
      <Sparkle size={22} className="empty-icon" />
      <strong>Not ready for questions yet</strong>
      <p>Tracer splits the papers into passages and prepares them for search. This takes a few minutes, once.</p>
      {running ? (
        <p><Loader2 size={14} className="spin" /> Preparing… <Link className="link-button" to={`/app/new?job=${running.id}`}>follow the steps</Link></p>
      ) : (
        <button className="primary-button" disabled={prepare.isPending} onClick={() => prepare.mutate()}>
          {prepare.isPending ? "Starting…" : "Prepare for questions"}
        </button>
      )}
      {prepare.error && <p className="error">{prepare.error.message}</p>}
    </div>
  );
}

function Conversation({ topicId, onOpen, prepared, papers }: Props & { prepared: number; papers: number }) {
  const [search, setSearch] = useSearchParams();
  const chatId = search.get("chat") ? Number(search.get("chat")) : null;
  const chats = useChats(topicId);
  const chat = useChat(chatId);
  const ask = useAsk(topicId);
  const remove = useDeleteChat();
  const toast = useToast();
  const [question, setQuestion] = useState("");
  const [live, setLive] = useState<Live | null>(null);        // the answer being streamed, until it is typed out
  const bottom = useRef<HTMLDivElement>(null);

  const openChat = (id: number | null) => {
    const next = new URLSearchParams(search);
    if (id) next.set("chat", String(id)); else next.delete("chat");
    setSearch(next);
  };
  // the saved question + answer stay hidden while the live copy is still typing them out
  const hidden = new Set(live?.result ? [live.result.question.id, live.result.answer.id] : []);
  const messages = (chat.data?.messages ?? []).filter((m) => !hidden.has(m.id));
  const scrollDown = () => bottom.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  // braces on purpose: newer browsers return a Promise from scrollIntoView, and an effect may only return a cleanup
  useEffect(() => { scrollDown(); }, [messages.length, live?.stage?.step, live?.sentences.length]);

  const send = (text: string) => {
    if (text.length < 3 || ask.isPending) return;
    setLive({ question: text, stage: null, sentences: [] });
    setQuestion("");
    ask.mutate({ question: text, chat_id: chatId, onEvent: (event, data) => setLive((l) => l && applyEvent(l, event, data)) }, {
      onSuccess: (result) => {
        if (result.chat_id !== chatId) openChat(result.chat_id);
        const parts = result.answer.answer?.parts ?? [];
        // the saved answer is the truth: the live copy finishes typing IT, then hands over to the saved message
        setLive((l) => (l && parts.length ? { ...l, result, sentences: parts.map((p) => ({ text: p.text, part: p })) } : null));
      },
      onError: () => { setLive(null); setQuestion(text); },   // give the question back to edit and retry
    });
  };
  const submit = (event: React.FormEvent) => { event.preventDefault(); send(question.trim()); };

  return (
    <div className="ask">
      <div className="ask-bar">
        <select className="input chat-select" value={chatId ?? ""} onChange={(e) => openChat(e.target.value ? Number(e.target.value) : null)}>
          <option value="">New chat</option>
          {(chats.data ?? []).map((c) => <option key={c.id} value={c.id}>{c.title}</option>)}
        </select>
        {chatId && (
          <>
            <button className="icon-button" title="New chat" onClick={() => openChat(null)}><Plus size={16} /></button>
            <button className="icon-button" title="Delete this chat"
                    onClick={() => remove.mutate(chatId, { onSuccess: () => { openChat(null); toast("Chat deleted"); } })}>
              <Trash2 size={16} />
            </button>
          </>
        )}
        <span className="muted ask-note">
          Answers use only these papers{prepared < papers ? ` · ${prepared} of ${papers} have text` : ""}
        </span>
      </div>

      <div className="thread">
        <div className="thread-inner">
          {!messages.length && !live && <Suggestions topicId={topicId} onPick={send} />}
          {messages.map((m) => m.role === "user"
            ? <div key={m.id} className="question">{m.content}</div>
            : <AnswerBlock key={m.id} message={m} onOpen={onOpen} />)}
          {live && (
            <>
              <div className="question">{live.question}</div>
              {live.sentences.some((s) => !s.dropped && s.text)
                ? <LiveAnswer live={live} onOpen={onOpen} onTyped={() => setLive(null)} onGrow={scrollDown} />
                : <ThinkingSteps stage={live.stage} />}
            </>
          )}
          {ask.error && <AskError error={ask.error} />}
          <div ref={bottom} />
        </div>
      </div>

      <form className="ask-form" onSubmit={submit}>
        <div className="composer">
          <textarea rows={1} maxLength={500} value={question} placeholder="Ask about these papers…"
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) submit(e); }} />
          <button className="send-button" disabled={question.trim().length < 3 || ask.isPending} aria-label="Ask">
            <ArrowUp size={16} />
          </button>
        </div>
        <p className="composer-hint">Every sentence cites a passage, and Tracer checks each quote against the paper.</p>
      </form>
    </div>
  );
}

/** First questions to try, from what this topic holds. */
function Suggestions({ topicId, onPick }: { topicId: number; onPick: (q: string) => void }) {
  const topic = useTopic(topicId).data;
  const first = topic?.papers?.[0];
  const ideas = [
    "What are the main findings across these papers?",
    (topic?.connections ?? []).some((c) => c.relation !== "agree") ? "Where do these papers disagree, and why?" : null,
    first ? `What did ${first.in_text.replace(/[()]/g, "")} find?` : null,
    "What limitations do the authors mention?",
  ].filter(Boolean) as string[];
  return (
    <div className="suggestions">
      <h2>Ask these papers anything</h2>
      <p className="muted">Answers come only from this topic's papers, with the exact passage for every claim.</p>
      <div className="suggestion-list">
        {ideas.map((q) => <button key={q} className="suggestion" onClick={() => onPick(q)}>{q}</button>)}
      </div>
    </div>
  );
}

const STEPS = ["searching", "reading", "writing", "checking"] as const;

/** The answer's steps as the server reports them (streamed), with its real numbers. Time shows on the step
 *  that is running, so a slow LLM looks slow instead of pretending. */
function ThinkingSteps({ stage }: { stage: AskStage | null }) {
  const at = stage ? STEPS.indexOf(stage.step) : 0;     // before the first event: the search is starting
  const seconds = useSeconds(at);
  const label = (step: (typeof STEPS)[number]) => {
    if (step === "searching") return stage?.step === "searching" ? `Searching ${stage.papers} papers` : "Searching your papers";
    if (step === "reading") {
      if (stage?.step !== "reading") return "Reading the best passages";
      return stage.passages ? `Reading ${stage.passages} passages from ${stage.papers} ${stage.papers === 1 ? "paper" : "papers"}`
                            : "No passage matches the question";
    }
    return step === "writing" ? "Writing the answer" : "Checking every quote";
  };
  return (
    <ol className="thinking" aria-live="polite">
      {STEPS.map((step, i) => (
        <li key={step} className={i < at ? "done" : i === at ? "now" : ""}>
          {i < at ? <Check size={13} /> : i === at ? <Loader2 size={13} className="spin" /> : <span className="dot" />}
          {" "}{label(step)}
          {i === at && seconds >= 2 && <span className="thinking-time">{seconds} s</span>}
        </li>
      ))}
    </ol>
  );
}

/** Whole seconds since `key` last changed. */
function useSeconds(key: unknown): number {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    setSeconds(0);
    const started = Date.now();
    const timer = setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 500);
    return () => clearInterval(timer);
  }, [key]);
  return seconds;
}

// ---------- the answer while it streams (SSE step 5) ----------

/** The streamed answer, typed out smoothly. Calls onTyped once the saved answer has arrived AND is fully shown. */
function LiveAnswer({ live, onOpen, onTyped, onGrow }: { live: Live; onOpen: OpenPaper; onTyped: () => void; onGrow: () => void }) {
  const visible = live.sentences.filter((s) => !s.dropped && s.text);
  const parts: AnswerPart[] = visible.map((s) => s.part ?? { text: s.text, flags: [] });
  const total = typedLength(parts);
  const shown = useTypewriter(total);
  const typed = shown >= total;
  useEffect(() => { if (typed && live.result) onTyped(); }, [typed, live.result]);   // eslint-disable-line
  useEffect(() => { onGrow(); }, [Math.floor(shown / 120)]);                       // eslint-disable-line
  const answer = { parts, answered: true, passages_searched: 0 } as Answer;
  return <AnswerView answer={answer} onOpen={onOpen} live={{ shown, checked: visible.map((s) => !!s.part) }} />;
}

/** How many characters to show: catches up with `total` a few characters per frame, faster when far behind.
 *  Gemini sends ~14 tokens per piece (measured in step 3): without this the text would jump in blocks. */
function useTypewriter(total: number): number {
  const reduced = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  const [shown, setShown] = useState(0);
  const target = useRef(total);
  target.current = total;
  useEffect(() => {
    if (reduced) return;
    let frame = 0;
    const tick = () => {
      setShown((s) => {
        const behind = target.current - s;
        return behind <= 0 ? s : Math.min(target.current, s + Math.max(2, Math.ceil(behind / 25)));
      });
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [reduced]);
  return reduced ? total : shown;
}

type Source = { n: number; citation: Citation; quotes: string[]; flagged: boolean };

/** Number the sources of an answer: one number per passage, in order of first use. */
function numberSources(parts: AnswerPart[]): { sources: Source[]; numberOf: (p: AnswerPart) => number | null } {
  const byChunk = new Map<number, Source>();
  for (const p of parts) {
    if (!p.citation) continue;
    const known = byChunk.get(p.citation.chunk_id);
    if (known) {
      if (!known.quotes.includes(p.citation.quote)) known.quotes.push(p.citation.quote);
      known.flagged ||= !!p.flags?.length;
    } else {
      byChunk.set(p.citation.chunk_id, { n: byChunk.size + 1, citation: p.citation, quotes: [p.citation.quote],
                                         flagged: !!p.flags?.length });
    }
  }
  return { sources: [...byChunk.values()], numberOf: (p) => (p.citation ? byChunk.get(p.citation.chunk_id)!.n : null) };
}

function AnswerBlock({ message, onOpen }: { message: ChatMessageInfo; onOpen: OpenPaper }) {
  if (!message.answer) return <div className="answer"><p>{message.content}</p></div>;
  return <AnswerView answer={message.answer} onOpen={onOpen} />;
}

/** An answer: saved (whole), or live (being typed out, `typing` set). Both look the same once typed. */
function AnswerView({ answer, onOpen, live }: { answer: Answer; onOpen: OpenPaper; live?: Typing }) {
  const toast = useToast();
  const shownParts = reveal(answer.parts ?? [], live);
  const parts = shownParts.map((s) => s.part);
  const { sources, numberOf } = numberSources(parts);
  const copy = async () => {
    const text = parts.map((p) => `${p.text}${numberOf(p) ? ` [${numberOf(p)}]` : ""}`).join(" ")
      + (sources.length ? "\n\n" + sources.map((s) => `[${s.n}] ${s.citation.label}: “${s.quotes[0]}”`).join("\n") : "");
    try { await navigator.clipboard.writeText(text); toast("Answer copied with its sources"); } catch { /* blocked */ }
  };

  return (
    <div className="answer" aria-busy={!!live}>
      {!answer.answered && !parts.length && (
        <p className="declined">These papers don't answer this{answer.missing ? `: ${answer.missing}` : "."}</p>
      )}
      <p className="answer-text">
        {parts.map((p, i) => {
          const n = numberOf(p);
          return (
            <span key={i} className={p.flags?.length ? "flagged" : undefined}>
              {shownParts[i].text}
              {n && p.citation && (
                <span className="cite-wrap">
                  <button className="cite-num" onClick={() => p.citation?.rank && onOpen(p.citation.rank, p.citation)}
                          aria-label={`Source ${n}: ${p.citation.label}`}>{n}</button>
                  <span className="cite-pop" role="tooltip">
                    <span className="cite-pop-head">{p.citation.label} · {p.citation.section}</span>
                    <span className="cite-pop-quote">“{p.citation.quote}”</span>
                    <span className="cite-pop-foot">Click to open the paper with this passage highlighted</span>
                  </span>
                </span>
              )}
              {!!p.flags?.length && <span className="flag-mark" title={p.flags.join("\n")}><AlertTriangle size={13} /></span>}{" "}
            </span>
          );
        })}
        {live && <span className="caret" aria-hidden />}
      </p>
      {answer.answered && answer.missing && <p className="muted small">Not covered: {answer.missing}</p>}
      {sources.length > 0 && (
        <div className="sources">
          {sources.map((s) => (
            <button key={s.n} className={s.flagged ? "source flagged-source" : "source"}
                    onClick={() => s.citation.rank && onOpen(s.citation.rank, s.citation)}>
              <span className="source-n">{s.n}</span>
              <span className="source-body">
                <span className="source-head">{s.citation.label} <span className="muted">· {s.citation.section}</span></span>
                <span className="source-quote">“{s.quotes[0]}”</span>
                {s.flagged && <span className="flag-text"><AlertTriangle size={12} /> a check failed: see the ⚠ in the answer</span>}
              </span>
            </button>
          ))}
        </div>
      )}
      {!live && (
        <div className="answer-actions">
          <button className="ghost-button" onClick={copy}><Copy size={13} /> Copy</button>
          {(answer.flags ?? []).map((f) => <span key={f} className="muted small">{f}</span>)}
        </div>
      )}
    </div>
  );
}

function AskError({ error }: { error: Error }) {
  const quota = error instanceof ApiError && error.status === 429;
  const raw = quota ? error.message : `Couldn't answer: ${error.message}`;
  const text = /[.!?]$/.test(raw) ? raw : `${raw}.`;
  return (
    <div className={quota ? "notice" : "notice notice-error"} role="alert">
      <AlertTriangle size={16} />
      <span>{text.charAt(0).toUpperCase() + text.slice(1)}{quota && <span className="muted"> Your question is still in the box below.</span>}</span>
    </div>
  );
}
