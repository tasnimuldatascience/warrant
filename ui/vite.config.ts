import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API mounts `UI_DIR/assets` at `/assets` and serves `UI_DIR/index.html` at `/`, and
// nothing else -- there is no SPA catch-all. So the build must emit into `assets/` (Vite's
// default) and the app must route on the hash, never on the path: `/timeline` would 404
// against the very server this bundle ships inside.
export default defineConfig({
  base: "/",
  plugins: [react()],
  build: {
    outDir: "dist",
    assetsDir: "assets",
    // Committed to git and read by a reviewer diffing a release. A sourcemap would triple
    // the tracked bytes for something no clone of this repo needs.
    sourcemap: false,
    target: "es2022",
  },
  server: {
    port: 5173,
    // `warrant serve` is a separate origin in dev. Proxying rather than pointing the client
    // at http://127.0.0.1:8000 keeps one code path: the fetch URLs are the same relative
    // ones the production bundle uses, so a path that works in dev works when mounted.
    proxy: Object.fromEntries(
      ["/api", "/health", "/ready", "/metrics"].map((p) => [
        p,
        { target: "http://127.0.0.1:8000", changeOrigin: false },
      ]),
    ),
  },
});
