import { FolderClosed, FolderX, Library, Loader2, LogOut, MessageSquare, PenSquare, Search } from "lucide-react";
import { useState } from "react";
import { Link, NavLink, useNavigate } from "react-router-dom";
import { Logo } from "../components/Logo";
import { SkeletonList } from "../components/Skeleton";
import { UsageMeter } from "../components/UsageMeter";
import { isActive, useChats, useJobs, useLogout, usePrefetchTopic, useTopics } from "../api/queries";
import { jobTitle } from "../lib/format";
import type { UserInfo } from "../api/types";

type Props = { user: UserInfo; open: boolean; onClose: () => void };

export function Sidebar({ user, open, onClose }: Props) {
  const [tab, setTab] = useState<"topics" | "chats" | "runs">("topics");
  const logout = useLogout();
  const navigate = useNavigate();

  return (
    <>
      {open && <div className="backdrop" onClick={onClose} />}
      <nav className={open ? "sidebar open" : "sidebar"}>
        <Link to="/app" className="side-brand"><Logo size={20} /></Link>

        <NavLink to="/app/new" className="nav-item"><PenSquare size={17} /> New research</NavLink>
        <NavLink to="/app" end className="nav-item"><Library size={17} /> Library</NavLink>
        <NavLink to="/app/search" className="nav-item"><Search size={17} /> Search</NavLink>

        <div className="side-tabs">
          <button className={tab === "topics" ? "side-tab active" : "side-tab"} onClick={() => setTab("topics")}>Topics</button>
          <button className={tab === "chats" ? "side-tab active" : "side-tab"} onClick={() => setTab("chats")}>Chats</button>
          <button className={tab === "runs" ? "side-tab active" : "side-tab"} onClick={() => setTab("runs")}>Runs</button>
        </div>
        <div className="side-list">{tab === "topics" ? <TopicList /> : tab === "chats" ? <ChatList /> : <RunList />}</div>
        <UsageMeter />
        <div className="account">
          <span className="avatar" aria-hidden="true">{(user.display_name || user.email)[0].toUpperCase()}</span>
          <span className="account-email" title={user.email}>{user.display_name || user.email}</span>
          <button className="icon-button" title="Log out"
                  onClick={() => logout.mutate(undefined, { onSettled: () => navigate("/login") })}>
            <LogOut size={16} />
          </button>
        </div>
      </nav>
    </>
  );
}

function TopicList() {
  const topics = useTopics();
  const jobs = useJobs();
  const prefetch = usePrefetchTopic();
  const running = (jobs.data ?? []).filter((j) => j.kind === "research" && isActive(j));

  if (topics.isPending) return <SkeletonList />;
  if (topics.error) return <div className="side-empty">{topics.error.message}</div>;
  return (
    <>
      {running.map((job) => (
        <NavLink key={`job-${job.id}`} to={`/app/new?job=${job.id}`} className="nav-item" title="Running">
          <Loader2 size={16} className="spin" />
          <span className="nav-label">{String(job.params.topic ?? "Research")}</span>
        </NavLink>
      ))}
      {topics.data.map((t) => (
        <NavLink key={t.id} to={`/app/topics/${t.id}`} className={t.has_results ? "nav-item" : "nav-item dim"}
                 title={`#${t.id} · ${t.query}${t.has_results ? "" : " (no results)"}`}
                 onMouseEnter={() => t.has_results && prefetch(t.id)}>
          {t.has_results ? <FolderClosed size={16} /> : <FolderX size={16} />}
          <span className="nav-label">{t.query}</span>
        </NavLink>
      ))}
      {!running.length && !topics.data.length && <div className="side-empty">No topics yet</div>}
    </>
  );
}

function RunList() {
  const jobs = useJobs();
  if (jobs.isPending) return <SkeletonList />;
  if (!jobs.data?.length) return <div className="side-empty">No runs yet</div>;
  return (
    <>
      {jobs.data.map((job) => (
        <NavLink key={job.id} to={`/app/new?job=${job.id}`} className="nav-item" title={job.status}>
          {isActive(job) ? <Loader2 size={16} className="spin" /> : <span className="nav-note" style={{ margin: 0, width: 16 }}>{job.status === "failed" ? "✕" : "✓"}</span>}
          <span className="nav-label">{jobTitle(job)}</span>
        </NavLink>
      ))}
    </>
  );
}

function ChatList() {
  const chats = useChats();
  if (chats.isPending) return <SkeletonList />;
  if (!chats.data?.length) return <div className="side-empty">No chats yet: open a topic and use “Ask”</div>;
  return (
    <>
      {chats.data.map((c) => (
        <NavLink key={c.id} to={`/app/topics/${c.topic_id}?view=ask&chat=${c.id}`} className="nav-item" title={c.title}>
          <MessageSquare size={16} />
          <span className="nav-label">{c.title}</span>
        </NavLink>
      ))}
    </>
  );
}
