import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The dev server forwards /api to the FastAPI backend, so the browser sees ONE origin:
// the login cookie just works and no CORS setup is needed.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://localhost:8010" },
    // the project lives on the Windows drive (/mnt/c); WSL can't see file-change events there
    watch: { usePolling: true, interval: 300 },
  },
});
