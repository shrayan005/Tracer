import { useEffect, useState } from "react";
import { Navigate, Outlet, useLocation } from "react-router-dom";
import { ApiError } from "../api/client";
import { useMe, useRefreshWhenJobsFinish } from "../api/queries";
import { ErrorBoundary } from "./ErrorBoundary";
import { Sidebar } from "./Sidebar";

/** What a page shows in the right-hand panel (null = no panel). Pages set it through the outlet context. */
export type PanelContent = React.ReactNode | null;
export type LayoutContext = { setPanel: (content: PanelContent) => void; openMenu: () => void };

/** The 3 panes. Every page inside needs a login: no session -> the login page (and back here after). */
export function AppLayout() {
  const me = useMe();
  const location = useLocation();
  const [panel, setPanel] = useState<PanelContent>(null);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => setMenuOpen(false), [location.pathname]); // close the phone menu after navigating

  if (me.isPending) return null; // a split second on first load; avoids a flash of the login page
  if (me.error instanceof ApiError && me.error.status === 401) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }
  if (me.error) return <div className="empty"><strong>Can't reach the server</strong>{me.error.message}</div>;

  return (
    <div className={panel ? "app" : "app app--no-panel"}>
      <Sidebar user={me.data} open={menuOpen} onClose={() => setMenuOpen(false)} />
      <main className="main">
        <ErrorBoundary resetKey={location.pathname}>
          <Outlet context={{ setPanel, openMenu: () => setMenuOpen(true) } satisfies LayoutContext} />
        </ErrorBoundary>
      </main>
      {panel && <aside className="panel">{panel}</aside>}
      <JobWatcher />
    </div>
  );
}

/** Rendered only when logged in (no 401 noise): refreshes things when a run finishes. */
function JobWatcher() {
  useRefreshWhenJobsFinish();
  return null;
}
