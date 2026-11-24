import { ArrowRight, BookOpenCheck, FileSearch, GitCompare, MessageSquareQuote, ShieldCheck } from "lucide-react";
import { Link } from "react-router-dom";
import { Logo } from "../components/Logo";
import "../styles/landing.css";

const STEPS = [
  { n: "01", icon: FileSearch, title: "Find", text: "Plans several searches for your question across open scholarly databases and merges the duplicates." },
  { n: "02", icon: ShieldCheck, title: "Screen and weigh", text: "Reads every abstract, keeps what is relevant and rates the evidence by its study design." },
  { n: "03", icon: BookOpenCheck, title: "Read in full", text: "Gets legal open full texts, pulls out methods and results, and checks how each study was done." },
  { n: "04", icon: GitCompare, title: "Connect", text: "Shows where papers agree, where they contradict each other, and what the evidence is still missing." },
  { n: "05", icon: MessageSquareQuote, title: "Ask", text: "Answers your questions from these papers only, with the exact passage behind every sentence." },
];

// Measured in the project's own evaluations (milestones 6-7): small test sets, see the note under the bars.
const RESULTS = [
  { label: "Answers cite the paper that holds the answer", value: 100, note: "11 questions" },
  { label: "Declines what the papers don't cover", value: 100, note: "no guessing" },
  { label: "Study-quality facts read correctly", value: 94, note: "10 full-text papers" },
  { label: "Relevant papers kept by screening", value: 90, note: "60 papers" },
  { label: "Connections between papers found", value: 90, note: "2 topics" },
];

const SOURCES = ["OpenAlex", "Europe PMC", "PubMed", "arXiv", "Semantic Scholar", "Unpaywall"];

const FAQ = [
  ["Where do the papers come from?", "From open scholarly databases: OpenAlex, Europe PMC (which includes PubMed), arXiv and Semantic Scholar. Full texts come only from legal open copies, found through Europe PMC, arXiv and Unpaywall."],
  ["What about paywalled papers?", "Tracer works from their abstract and says so on the paper. Quality checks that need the methods section are only made from full text."],
  ["Can it make things up?", "Answers are written only from passages of your papers. Code checks that every quoted passage and every number really appears in the paper; anything that fails is marked ⚠, not hidden. When the papers don't cover a question, Tracer says so."],
  ["Who can see my research?", "Your topics, papers and chats are visible to your account only. To answer questions, passages of the papers are sent to the language model provider (Google Gemini)."],
  ["How long does a topic take?", "About five minutes the first time: finding and screening take a minute or two, reading and preparing the papers for questions the rest. Everything is reused afterwards."],
];

/** The public front page: what Tracer does, shown with the real product and its measured results. */
export function LandingPage() {
  return (
    <div className="landing">
      <header className="l-nav">
        <div className="l-wrap l-nav-inner">
          <Link to="/" aria-label="Tracer home"><Logo size={22} /></Link>
          <nav className="l-links">
            <a href="#how">How it works</a>
            <a href="#proof">Shows its work</a>
            <a href="#results">Results</a>
            <a href="#faq">FAQ</a>
          </nav>
          <div className="l-actions">
            <Link to="/login" className="l-btn l-btn-ghost">Log in</Link>
            <Link to="/app" className="l-btn l-btn-dark">Start researching</Link>
          </div>
        </div>
      </header>

      <section className="l-hero">
        <div className="l-wrap">
          <p className="l-eyebrow">Literature research, with receipts</p>
          <h1>Research that shows<br /><span className="l-accent-text">its sources.</span></h1>
          <p className="l-sub">
            Tracer finds the best papers on your question, reads them in full where it can, and answers with
            the exact passage behind every sentence: checked by code, not taken on trust.
          </p>
          <div className="l-cta">
            <Link to="/app" className="l-btn l-btn-dark l-btn-lg">Start researching <ArrowRight size={16} /></Link>
            <a href="#how" className="l-btn l-btn-light l-btn-lg">See how it works</a>
          </div>
          <figure className="l-shot">
            <div className="l-shot-bar"><i /><i /><i /><span>tracer · creatine supplementation and muscle strength in older adults</span></div>
            <img src="/screenshots/app-papers.jpg" alt="Tracer showing the ten selected papers of a topic, with one paper's summary, trust rating and quotes in the side panel"
                 width={1440} height={900} />
          </figure>
        </div>
      </section>

      <section className="l-sources">
        <div className="l-wrap">
          <p>Searches the open scholarly record</p>
          <ul>{SOURCES.map((s) => <li key={s}>{s}</li>)}</ul>
        </div>
      </section>

      <section className="l-section" id="proof">
        <div className="l-wrap l-split">
          <div>
            <p className="l-eyebrow">Shows its work</p>
            <h2>Every sentence, traced to its passage.</h2>
            <ul className="l-points">
              <li><strong>Numbered citations</strong> open the paper with the quoted passage highlighted.</li>
              <li><strong>Checked by code:</strong> every quote and number is looked up in the paper; a failed check is marked, never hidden.</li>
              <li><strong>No guessing:</strong> when your papers don't cover a question, Tracer says so.</li>
            </ul>
          </div>
          <DemoAnswer />
        </div>
      </section>

      <section className="l-section l-soft" id="how">
        <div className="l-wrap">
          <p className="l-eyebrow">How it works</p>
          <h2>From a question to a cited answer.</h2>
          <div className="l-steps">
            {STEPS.map(({ n, icon: Icon, title, text }) => (
              <article key={n} className="l-step">
                <div className="l-step-top"><span className="l-step-n">{n}</span><Icon size={18} /></div>
                <h3>{title}</h3>
                <p>{text}</p>
              </article>
            ))}
          </div>
          <figure className="l-shot l-shot-small">
            <div className="l-shot-bar"><i /><i /><i /><span>tracer · ask</span></div>
            <img src="/screenshots/app-ask.jpg" alt="Asking a topic's papers a question: the answer cites two papers with numbered sources and their quotes"
                 width={1440} height={900} loading="lazy" />
          </figure>
        </div>
      </section>

      <section className="l-section" id="results">
        <div className="l-wrap l-split">
          <div>
            <p className="l-eyebrow">Measured, not claimed</p>
            <h2>We test Tracer on frozen research topics.</h2>
            <p className="l-muted">
              Each part of the pipeline has its own evaluation, re-run after every change. These are the latest
              results.
            </p>
            <p className="l-fine">
              Small test sets (two topics); the reference labels were written with an AI and marked as such.
              Read them as direction, not as a benchmark.
            </p>
          </div>
          <div className="l-bars">
            {RESULTS.map((r) => (
              <div key={r.label} className="l-bar">
                <div className="l-bar-head"><span>{r.label}</span><strong>{r.value}%</strong></div>
                <div className="l-bar-track"><div className="l-bar-fill" style={{ width: `${r.value}%` }} /></div>
                <span className="l-bar-note">{r.note}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="l-section l-soft" id="faq">
        <div className="l-wrap l-faq">
          <h2>Questions</h2>
          {FAQ.map(([q, a]) => (
            <details key={q}>
              <summary>{q}</summary>
              <p>{a}</p>
            </details>
          ))}
        </div>
      </section>

      <section className="l-final">
        <div className="l-wrap">
          <h2>Start with a question.</h2>
          <p className="l-sub">Tracer does the searching, reading and cross-checking. You get the evidence, with its sources.</p>
          <Link to="/app" className="l-btn l-btn-light l-btn-lg">Start researching <ArrowRight size={16} /></Link>
        </div>
      </section>

      <footer className="l-footer">
        <div className="l-wrap l-footer-inner">
          <div><Logo size={18} /><p>Traces every claim back to its source.</p></div>
          <nav><a href="#how">How it works</a><a href="#results">Results</a><a href="#faq">FAQ</a><Link to="/login">Log in</Link></nav>
          <p className="l-copy">© 2026 Tracer</p>
        </div>
      </footer>
    </div>
  );
}

/** A cited answer as Tracer shows it (the real answer from the creatine topic). */
function DemoAnswer() {
  return (
    <div className="l-demo">
      <div className="l-demo-q">Did any study find no effect of creatine on muscle mass?</div>
      <p className="l-demo-a">
        Yao et al. found that creatine combined with resistance training showed no clear additional benefit for
        muscle mass<sup className="l-num">1</sup>. In contrast, Chilibeck et al. reported greater increases in lean
        tissue mass with creatine<sup className="l-num">2</sup>.
      </p>
      <div className="l-demo-sources">
        <div className="l-demo-source"><span className="l-num">1</span><div><b>Yao et al., 2026</b> · discussion<br />
          <q>No clear additional benefits were observed for muscle mass or physical function.</q></div></div>
        <div className="l-demo-source"><span className="l-num">2</span><div><b>Chilibeck et al., 2017</b> · abstract<br />
          <q>Creatine supplementation resulted in greater increases in lean tissue mass</q></div></div>
      </div>
      <div className="l-demo-check"><ShieldCheck size={14} /> Both quotes found in the papers</div>
    </div>
  );
}
