import { defineConfig } from "vite";

// `npm run dev` serves the page and passes what it asks of the server on to webapp.py.
const server = "http://127.0.0.1:8000";

export default defineConfig({
  root: "web",
  build: { outDir: "dist", emptyOutDir: true },
  server: { proxy: { "/api": server, "/media": server } },
});
