import { FileText } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useSearch } from "../api/queries";
import { Topbar } from "../layout/Topbar";

/** Search passages across all your papers (meaning + keywords). No LLM: free and quick. */
export function SearchPage() {
  const [params, setParams] = useSearchParams();
  const [text, setText] = useState(params.get("q") ?? "");
  const query = params.get("q") ?? "";
  const results = useSearch(query);

  useEffect(() => {                                   // search 400 ms after typing stops (and keep it in the URL)
    const timer = setTimeout(() => {
      if (text.trim() !== query) setParams(text.trim() ? { q: text.trim() } : {}, { replace: true });
    }, 400);
    return () => clearTimeout(timer);
  }, [text, query, setParams]);

  return (
    <>
      <Topbar title="Search" />
      <div className="main-body">
        <div className="search-page">
          <input className="input" autoFocus placeholder="Search your papers, e.g. “HMB” or “dose per day”"
                 value={text} onChange={(e) => setText(e.target.value)} />
          {results.isFetching && <p className="muted">Searching…</p>}
          {results.error && <p className="error">{results.error.message}</p>}
          {results.data && !results.data.length && <p className="muted">Nothing found in your papers.</p>}
          <ul className="plain results">
            {(results.data ?? []).map((hit, i) => (
              <li key={i} className="result">
                <Link to={`/app/topics/${hit.topic_id}/papers/${hit.rank}`} className="result-title">
                  <FileText size={15} /> <span>{hit.title}</span>
                </Link>
                <div className="muted result-meta">{hit.label} · {hit.section}</div>
                <p className="snippet"><Highlighted text={hit.snippet} /></p>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </>
  );
}

/** The backend marks matched words with \u0002 … \u0003: shown bold, safely (no HTML from the server). */
function Highlighted({ text }: { text: string }) {
  const parts = text.split(/\u0002|\u0003/);
  return <>{parts.map((part, i) => (i % 2 ? <mark key={i}>{part}</mark> : <span key={i}>{part}</span>))}…</>;
}
