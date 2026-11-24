import { Link } from "react-router-dom";
import { Topbar } from "../layout/Topbar";

export function NotFoundPage() {
  return (
    <>
      <Topbar title="Not found" />
      <div className="empty"><strong>This page doesn't exist</strong><Link to="/app" className="link-button">Back to the library</Link></div>
    </>
  );
}
