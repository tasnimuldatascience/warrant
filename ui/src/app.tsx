import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api, type Meta, type Ready } from "./api";
import { AskProvider } from "./ask";
import { SCREENS, num, stored, store, today, useAsync, useRoute, type Screen } from "./lib";
import { Failure, Note } from "./ui";
import Ask from "./screens/Ask";
import Timeline from "./screens/Timeline";
import Diff from "./screens/Diff";
import Trace from "./screens/Trace";

// -- corpus context ------------------------------------------------------------------------

export interface Corpus {
  meta: Meta;
  ready: Ready | null;
  /** Clamped to the corpus range, so the date controls can never ask about a day it cannot answer. */
  clampDate: (iso: string) => string;
}

const CorpusCtx = createContext<Corpus | null>(null);

export function useCorpus(): Corpus {
  const c = useContext(CorpusCtx);
  if (!c) throw new Error("useCorpus outside provider");
  return c;
}

// -- chrome --------------------------------------------------------------------------------

const RAIL: { screen: Screen; n: string; what: string }[] = [
  { screen: "ask", n: "01", what: "A question, a date, a scope — and the evidence first." },
  { screen: "timeline", n: "02", what: "Every version of a section, and which one was law." },
  { screen: "diff", n: "03", what: "What the words actually became." },
  { screen: "trace", n: "04", what: "Which stage is answerable for the answer." },
];

type Theme = "system" | "light" | "dark";

function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    const v = stored("warrant.theme");
    return v === "light" || v === "dark" ? v : "system";
  });
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    store("warrant.theme", theme);
  }, [theme]);
  const cycle = useCallback(
    () => setTheme((t) => (t === "system" ? "light" : t === "light" ? "dark" : "system")),
    [],
  );
  return [theme, cycle];
}

function Lamp({ ready, onRefresh }: { ready: Ready | null; onRefresh: () => void }) {
  if (!ready) {
    return (
      <button className="lamp lamp--wait" onClick={onRefresh}>
        <span className="lamp__dot" />
        checking
      </button>
    );
  }
  // Three distinguishable states, because they call for three different actions: serving,
  // serving degraded, and not serving. Collapsing them into "up/down" is what makes a
  // readiness endpoint useless.
  const degraded = ready.ready && (ready.uncovered_chunks ?? 0) > 0;
  const kind = !ready.ready ? "bad" : degraded ? "warn" : "ok";
  const text = !ready.ready ? "not ready" : degraded ? "degraded" : "ready";
  return (
    <button
      className={`lamp lamp--${kind}`}
      onClick={onRefresh}
      title={ready.detail ?? "re-check /ready"}
    >
      <span className="lamp__dot" />
      {text}
    </button>
  );
}

function Masthead({
  meta,
  ready,
  onRefresh,
}: {
  meta: Meta | null;
  ready: Ready | null;
  onRefresh: () => void;
}) {
  const [theme, cycleTheme] = useTheme();
  const facts: [string, string][] = meta
    ? [
        ["chunk versions", num(meta.chunks)],
        ["sections", num(meta.sections)],
        ["cfr parts", num(meta.part_count)],
        ["answerable", `${meta.earliest ?? "—"} → ${meta.latest ?? "—"}`],
        ["final k", String(meta.final_k)],
        ["config", meta.config_hash],
      ]
    : [];
  return (
    <header className="masthead">
      <div className="masthead__mark">
        <div className="masthead__title">
          warrant<span className="dot">.</span>
        </div>
        <div className="masthead__sub">5 CFR · bitemporal retrieval</div>
      </div>
      <div className="masthead__facts">
        <div className="scroller">
          {facts.map(([k, v]) => (
            <div className="fact" key={k}>
              <span className="fact__k">{k}</span>
              <span className="fact__v">{v}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="masthead__tools">
        <Lamp ready={ready} onRefresh={onRefresh} />
        <button
          className="icon-btn"
          onClick={cycleTheme}
          title={`theme: ${theme} — click to change`}
          aria-label={`Colour theme: ${theme}. Activate to change.`}
        >
          <span aria-hidden="true" className="mono" style={{ fontSize: "0.7rem" }}>
            {theme === "system" ? "◐" : theme === "light" ? "☼" : "☾"}
          </span>
        </button>
      </div>
    </header>
  );
}

// -- readiness banner ------------------------------------------------------------------------

function ReadyBanner({ ready }: { ready: Ready }) {
  if (!ready.ready) {
    return (
      <div style={{ marginBottom: "1.5rem" }}>
        <Note
          kind="warn"
          label="not ready"
          title={
            ready.corpus
              ? "The corpus is loaded but the models are not built yet."
              : "No corpus. Retrieval cannot run."
          }
          detail={ready.detail ?? undefined}
        >
          <pre className="cmd">
            {ready.corpus
              ? "# models build on the first request, or eagerly at startup:\nmake serve"
              : "make fetch && make build && make index"}
          </pre>
          <p className="hint" style={{ marginTop: "0.6rem" }}>
            {ready.corpus
              ? "Started with --no-warm, the first ask pays the model construction itself: a " +
                "cold SentenceTransformer is 127 MB and a cold generator 2,944 MB, so expect " +
                "the first dense timing to be seconds rather than milliseconds."
              : "Every screen below will refuse rather than invent. That is the intended behaviour."}
          </p>
        </Note>
      </div>
    );
  }
  if (ready.uncovered_chunks === null) {
    return (
      <div style={{ marginBottom: "1.5rem" }}>
        <Note
          kind="plain"
          label="lexical only"
          title="No dense index is loaded, so retrieval is BM25 alone."
          detail="uncovered_chunks: null — nothing to be missing from, which is not the same as zero."
        >
          <p className="hint" style={{ marginTop: "0.5rem" }}>
            On the measured bucket this costs nothing: lexical-only is 14.1 ms p50 at the same
            96.7% sufficiency as lexical + dense. Fusion, however, is fusing one list.
          </p>
        </Note>
      </div>
    );
  }
  if (ready.uncovered_chunks > 0) {
    return (
      <div style={{ marginBottom: "1.5rem" }}>
        <Note
          kind="warn"
          label="degraded"
          title={`${num(ready.uncovered_chunks)} believed chunks have no vector.`}
          detail="They are still found by BM25, so they appear in one of the two fused rank lists instead of two, lose roughly half their fused score, and never disappear outright."
        >
          <pre className="cmd">warrant index build</pre>
        </Note>
      </div>
    );
  }
  return null;
}

// -- app -------------------------------------------------------------------------------------

export default function App() {
  const [route, go] = useRoute();
  const [meta, reloadMeta] = useAsync((s) => api.meta(s), []);
  const [ready, reloadReady] = useAsync((s) => api.ready(s), []);

  // Poll readiness while it is not ready, and stop once it is. A dashboard that keeps
  // hammering an endpoint it already has the answer from is noise in the very metrics this
  // project publishes.
  const readyValue = ready.state === "ok" ? ready.value : null;
  useEffect(() => {
    if (readyValue?.ready) return;
    const id = window.setInterval(reloadReady, 4000);
    return () => window.clearInterval(id);
  }, [readyValue?.ready, reloadReady]);

  // A corpus that appears after the fact (models finished warming) should refresh the facts.
  const corpusPresent = readyValue?.corpus;
  useEffect(() => {
    if (corpusPresent && meta.state === "failed") reloadMeta();
  }, [corpusPresent, meta.state, reloadMeta]);

  const metaValue = meta.state === "ok" ? meta.value : null;
  const corpus = useMemo<Corpus | null>(() => {
    if (!metaValue) return null;
    const lo = metaValue.earliest ?? metaValue.history_floor;
    const hi = metaValue.latest ?? today();
    return {
      meta: metaValue,
      ready: readyValue,
      clampDate: (iso: string) => (iso < lo ? lo : iso > hi ? hi : iso),
    };
  }, [metaValue, readyValue]);

  return (
    <div className="shell">
      <a className="skip" href="#main">
        Skip to content
      </a>
      <Masthead meta={metaValue} ready={readyValue} onRefresh={reloadReady} />
      <div className="body">
        <nav className="rail" aria-label="Screens">
          <ul className="rail__list">
            {RAIL.map((item) => (
              <li key={item.screen}>
                <button
                  className="rail__link"
                  aria-current={route.screen === item.screen ? "page" : undefined}
                  onClick={() => go(item.screen)}
                >
                  <span className="rail__n">{item.n}</span>
                  <span className="rail__name">{item.screen}</span>
                  <span className="rail__what">{item.what}</span>
                </button>
              </li>
            ))}
          </ul>
          <div className="rail__foot">
            Applicability, never authorization. eCFR is published law; filtering by who is
            asking is a correctness question, not a security boundary.
          </div>
        </nav>
        <main className="main" id="main" tabIndex={-1}>
          {meta.state === "failed" ? (
            <div className="screen">
              <div className="screen__head">
                <h1 className="screen__title">No corpus.</h1>
                <p className="screen__lede">
                  The API is reachable but has nothing to read. Nothing below can be
                  fabricated from an empty store, so nothing below is shown.
                </p>
              </div>
              <Failure error={meta.error} onRetry={reloadMeta} />
              <pre className="cmd" style={{ marginTop: "1rem" }}>
                {"make fetch    # eCFR point-in-time snapshots (cached, ~10 min once)\n" +
                  "make build    # parse into the bitemporal store\n" +
                  "make index    # embed for dense retrieval\n" +
                  "make serve    # the API on :8000"}
              </pre>
            </div>
          ) : !corpus ? (
            <div className="screen">
              <p className="label">loading the corpus index…</p>
            </div>
          ) : (
            <CorpusCtx.Provider value={corpus}>
              <AskProvider>
                {readyValue ? <ReadyBanner ready={readyValue} /> : null}
                {route.screen === "ask" ? <Ask /> : null}
                {route.screen === "timeline" ? <Timeline route={route} go={go} /> : null}
                {route.screen === "diff" ? <Diff route={route} go={go} /> : null}
                {route.screen === "trace" ? <Trace go={go} /> : null}
              </AskProvider>
            </CorpusCtx.Provider>
          )}
        </main>
      </div>
    </div>
  );
}

export { SCREENS };
