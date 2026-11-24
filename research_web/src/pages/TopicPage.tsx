import { Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate, useOutletContext, useParams, useSearchParams } from "react-router-dom";
import { ApiError } from "../api/client";
import { isActive, useAnalyze, useJobs, useTopic } from "../api/queries";
import { PaperPanel } from "../components/PaperPanel";
import { PaperTable } from "../components/PaperTable";
import { AskView } from "../components/AskView";
import { SkeletonRows } from "../components/Skeleton";
import type { Citation } from "../api/types";
import { ConnectionsView, GapsView, OverviewView } from "../components/ReportViews";
import { TopicTabs } from "../components/TopicTabs";
import type { LayoutContext } from "../layout/AppLayout";
import { useOpenTopics } from "../lib/openTopics";

const VIEWS = ["papers", "connections", "gaps", "overview", "ask"] as const;
type View = (typeof VIEWS)[number];

/** One topic: tabs of open topics, four views, and the selected paper in the right panel.
 *  Everything comes from ONE request (GET /api/topics/{id}); switching views and papers is instant. */
export function TopicPage() {
  const params = useParams();
  const id = Number(params.topicId);
  const rank = params.rank ? Number(params.rank) : null;
  const [search, setSearch] = useSearchParams();
  const view: View = VIEWS.includes(search.get("view") as View) ? (search.get("view") as View) : "papers";
  const navigate = useNavigate();
  const { setPanel } = useOutletContext<LayoutContext>();
  const { open } = useOpenTopics();
  const topic = useTopic(id);
  const report = topic.data;

  useEffect(() => { if (report) open({ id: report.topic_id, title: report.topic }); }, [report, open]);

  const suffix = search.toString() ? `?${search}` : "";          // keep the view (and the open chat)
  // a citation clicked in Ask: the panel shows its passage with the quote highlighted
  const [highlight, setHighlight] = useState<Citation | null>(null);
  const openPaper = (r: number, citation?: Citation) => {
    setHighlight(citation ?? null);
    navigate(`/app/topics/${id}/papers/${r}${suffix}`);
  };

  const selected = report?.papers.find((p) => p.rank === rank) ?? null;
  // Only when the paper (or the data) changes: setting the panel re-renders the layout, so running this
  // on every render would loop forever.
  useEffect(() => {
    const go = (path: string) => navigate(path);
    setPanel(report && selected
      ? <PaperPanel report={report} paper={selected} onClose={() => go(`/app/topics/${id}${suffix}`)}
                    onOpen={(r) => go(`/app/topics/${id}/papers/${r}${suffix}`)}
                    highlight={highlight && highlight.rank === selected.rank ? highlight : null} />
      : null);
  }, [report, selected, id, suffix, navigate, setPanel, highlight]);
  useEffect(() => () => setPanel(null), [setPanel]); // leaving the page closes the panel

  return (
    <>
      <TopicTabs activeId={id} />
      {topic.isPending && <SkeletonRows rows={8} columns={4} />}
      {topic.error && <TopicError error={topic.error} />}
      {report && (
        <>
          <div className="viewbar">
            {VIEWS.map((v) => (
              <button key={v} className={v === view ? "view-tab active" : "view-tab"}
                      onClick={() => setSearch(v === "papers" ? {} : { view: v })}>
                {v[0].toUpperCase() + v.slice(1)}
                {v === "papers" && <span className="count">{report.papers.length}</span>}
                {v === "connections" && !!report.connections?.length && <span className="count">{report.connections.length}</span>}
              </button>
            ))}
            <span className="spacer" />
            <AnalyseButton topicId={id} analysed={!!report.gaps} />
          </div>
          <div className="main-body">
            {view === "papers" && <PaperTable papers={report.papers} selected={rank} onSelect={openPaper} />}
            {view === "connections" && <ConnectionsView report={report} onOpen={openPaper} />}
            {view === "gaps" && <GapsView report={report} />}
            {view === "overview" && <OverviewView report={report} />}
            {view === "ask" && <AskView topicId={id} onOpen={openPaper} />}
          </div>
        </>
      )}
    </>
  );
}

function TopicError({ error }: { error: Error }) {
  const notReady = error instanceof ApiError && error.status === 409;
  return (
    <div className="empty">
      <strong>{notReady ? "No results yet" : "Can't open this topic"}</strong>
      {notReady ? "This research is still running, or it failed (see Runs)." : error.message}
    </div>
  );
}

function AnalyseButton({ topicId, analysed }: { topicId: number; analysed: boolean }) {
  const analyse = useAnalyze(topicId);
  const jobs = useJobs();
  const running = (jobs.data ?? []).some((j) => j.kind === "analyze" && j.params.topic_id === topicId && isActive(j));
  if (running || analyse.isPending) {
    return <span className="muted nowrap"><Loader2 size={14} className="spin" /> Analysing…</span>;
  }
  return (
    <span className="nowrap">
      {analyse.error && <span className="error-inline">{analyse.error.message}</span>}
      <button className="small-button" onClick={() => analyse.mutate(analysed)}
              title={analysed ? "Analyse again (claims, quality, connections, gaps)" : "Find claims, quality, connections and gaps"}>
        {analysed ? "Re-analyse" : "Analyse"}
      </button>
    </span>
  );
}
