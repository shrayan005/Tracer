import { FolderClosed, FolderX } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { usePrefetchTopic, useTopics } from "../api/queries";
import { Topbar } from "../layout/Topbar";
import { SkeletonRows } from "../components/Skeleton";
import { relativeDate } from "../lib/format";

const date = new Intl.DateTimeFormat(undefined, { dateStyle: "medium" });

/** All your topics, as a table (like the file list in the screenshot). */
export function LibraryPage() {
  const topics = useTopics();
  const navigate = useNavigate();
  const prefetch = usePrefetchTopic();

  return (
    <>
      <Topbar title="Library" />
      <div className="main-body">
        {topics.isPending && <SkeletonRows rows={6} columns={4} />}
        {topics.error && <div className="empty"><strong>Couldn't load your topics</strong>{topics.error.message}</div>}
        {topics.data?.length === 0 && (
          <div className="empty"><strong>No topics yet</strong>Start one with “New research”: Tracer finds, reads and connects the papers for you.</div>
        )}
        {!!topics.data?.length && (
          <table className="table">
            <thead>
              <tr><th>Name</th><th className="hide-small">Field</th><th>Papers</th><th className="hide-small">Added</th></tr>
            </thead>
            <tbody>
              {topics.data.map((t) => (
                <tr key={t.id} style={{ cursor: t.has_results ? "pointer" : "default" }}
                    onMouseEnter={() => t.has_results && prefetch(t.id)}
                    onClick={() => t.has_results && navigate(`/app/topics/${t.id}`)}>
                  <td>
                    <div className="name">
                      {t.has_results ? <FolderClosed size={16} /> : <FolderX size={16} className="muted" />}
                      <span className={t.has_results ? undefined : "muted"}>{t.query}</span>
                      <span className="muted topic-id">#{t.id}</span>
                    </div>
                  </td>
                  <td className="muted hide-small">{t.field ?? "—"}</td>
                  <td className="nowrap">{t.has_results ? t.papers : <span className="muted">no results</span>}</td>
                  <td className="muted nowrap hide-small" title={date.format(new Date(t.created_at))}>{relativeDate(t.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
