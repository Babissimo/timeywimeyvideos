import { defineConfig } from "vite";

export default defineConfig({
  root: "web",
  // Relative addresses, so the built page works from any folder of a static server.
  base: "./",
  build: { outDir: "dist", emptyOutDir: true },
});
