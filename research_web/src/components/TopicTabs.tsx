import { FolderOpen, Menu, X } from "lucide-react";
import { NavLink, useNavigate, useOutletContext } from "react-router-dom";
import type { LayoutContext } from "../layout/AppLayout";
import { useOpenTopics } from "../lib/openTopics";

/** The open topics as tabs (like the screenshot's "Partial Reprogram… ×"). */
export function TopicTabs({ activeId }: { activeId: number }) {
  const { topics, close } = useOpenTopics();
  const { openMenu } = useOutletContext<LayoutContext>();
  const navigate = useNavigate();

  const closeTab = (id: number) => {
    const index = topics.findIndex((t) => t.id === id);
    close(id);
    if (id !== activeId) return;
    const next = topics[index + 1] ?? topics[index - 1];          // the neighbour, like a browser
    navigate(next && next.id !== id ? `/app/topics/${next.id}` : "/app");
  };

  return (
    <div className="tabs">
      <button className="menu-button" onClick={openMenu} aria-label="Menu"><Menu size={18} /></button>
      {topics.map((t) => (
        <div key={t.id} className={t.id === activeId ? "tab active" : "tab"}>
          <NavLink to={`/app/topics/${t.id}`} className="tab-link" title={t.title}>
            <FolderOpen size={15} /><span>{t.title}</span>
          </NavLink>
          <button className="tab-close" aria-label={`Close ${t.title}`} onClick={() => closeTab(t.id)}>
            <X size={14} />
          </button>
        </div>
      ))}
    </div>
  );
}
