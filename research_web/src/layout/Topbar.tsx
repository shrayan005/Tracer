import { Menu } from "lucide-react";
import { useOutletContext } from "react-router-dom";
import type { LayoutContext } from "./AppLayout";

/** The top line of the main pane: the page title, and the menu button on phones. */
export function Topbar({ title, children }: { title: React.ReactNode; children?: React.ReactNode }) {
  const { openMenu } = useOutletContext<LayoutContext>();
  return (
    <header className="topbar">
      <button className="menu-button" onClick={openMenu} aria-label="Menu"><Menu size={18} /></button>
      <h1>{title}</h1>
      {children}
    </header>
  );
}
