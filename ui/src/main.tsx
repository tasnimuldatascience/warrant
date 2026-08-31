import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import App from "./app";

const root = document.getElementById("root");
if (!root) throw new Error("no #root; index.html and this entry disagree");

// StrictMode double-invokes effects in development, which is the point: `useAsync` aborts the
// superseded request on cleanup, and an abort that is not handled shows up here as a spurious
// "failed" state rather than in production, months later, on a slow connection.
createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
