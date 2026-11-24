import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { AppLayout } from "./layout/AppLayout";
import { LandingPage } from "./pages/LandingPage";
import { LibraryPage } from "./pages/LibraryPage";
import { LoginPage } from "./pages/LoginPage";
import { NewResearchPage } from "./pages/NewResearchPage";
import { NotFoundPage } from "./pages/NotFoundPage";
import { SearchPage } from "./pages/SearchPage";
import { TopicPage } from "./pages/TopicPage";

export function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/app" element={<AppLayout />}>
        <Route index element={<LibraryPage />} />
        <Route path="new" element={<NewResearchPage />} />
        <Route path="search" element={<SearchPage />} />
        <Route path="topics/:topicId" element={<TopicPage />} />
        <Route path="topics/:topicId/papers/:rank" element={<TopicPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
      {/* links from before the landing page (7.8): /topics/12 -> /app/topics/12 */}
      <Route path="/topics/*" element={<ToApp />} />
      <Route path="/new" element={<ToApp />} />
      <Route path="/search" element={<ToApp />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

function ToApp() {
  const { pathname, search } = useLocation();
  return <Navigate to={`/app${pathname}${search}`} replace />;
}
