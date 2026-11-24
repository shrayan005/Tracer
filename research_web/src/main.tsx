import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { ApiError } from "./api/client";
import { App } from "./App";
import "@fontsource-variable/inter";
import "./styles/app.css";
import { ToastProvider } from "./components/Toast";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000, // a page you just saw opens instantly from the cache, then refreshes quietly
      refetchOnWindowFocus: false,
      // retrying can't fix "not logged in", "not yours" or "not ready yet": only network/server hiccups
      retry: (count, error) => !(error instanceof ApiError && error.status < 500) && count < 2,
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <ToastProvider>
          <App />
        </ToastProvider>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
