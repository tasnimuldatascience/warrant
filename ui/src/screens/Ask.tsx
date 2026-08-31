import { useEffect, useMemo, useState } from "react";
import { useCorpus } from "../app";
import { useAsk, type AskState } from "../ask";
import type { StreamEvidence } from "../api";
import {
  AUTHORITY,
  badDate,
  inForce,
  longDate,
  ms,
  num,
  sectionOf,
  span,
  store,
  stored,
  superseded,
  today,
  useNow,
} from "../lib";
import { Chips, Copy, Empty, Field, Note, Refusal, SectionLabel, Stamp, Tag } from "../ui";

/**
 * Ask.
 *
 * The whole screen is built around one measured fact: retrieval returns in ~18 ms and
 * generation takes ~7 s. So there is no single spinner. The evidence ledger is populated the
 * instant the `evidence` frame lands and is never replaced; the prose arrives underneath it,
 * later, as validated claims. A reader who only wanted the regulation has it in a fiftieth of
 * a second and can leave.
 */

const EXAMPLES = [
  "How long is the probationary period for a new federal employee?",
  "How much annual leave does an employee earn each pay period?",
  "When may an agency deny a within-grade increase?",
  "How long can a term appointment last?",
];

export default function Ask() {
  const { meta, ready, clampDate } = useCorpus();
  const { state, run, cancel, reset } = useAsk();

  const [q, setQ] = useState(() => stored("warrant.q") ?? EXAMPLES[0]);
  const [asOf, setAsOf] = useState(() =>
    clampDate(stored("warrant.asOf") ?? meta.latest ?? today()),
  );
  const [pay, setPay] = useState<string | null>(() => stored("warrant.pay"));
  const [service, setService] = useState<string | null>(() => stored("warrant.service"));

  useEffect(() => {
    store("warrant.q", q);
    store("warrant.asOf", asOf);
    store("warrant.pay", pay ?? "");
    store("warrant.service", service ?? "");
  }, [q, asOf, pay, service]);

  const dateProblem = badDate(asOf);
  const outOfRange =
    !dateProblem && meta.earliest && meta.latest && (asOf < meta.earliest || asOf > meta.latest);
  const tooShort = q.trim().length < 2;
  const running =
    state.phase === "opening" || state.phase === "retrieved" || state.phase === "generating";
  const blocked = !ready?.corpus;

  function submit(e?: React.FormEvent) {
    e?.preventDefault();
    if (dateProblem || tooShort || running || blocked) return;
    run({ q: q.trim(), as_of: asOf, pay_system: pay, service });
  }

  const payValues = meta.facets.pay_system ?? [];
  const serviceValues = meta.facets.service ?? [];

  return (
    <div className="screen">
      <div className="screen__head">
        <h1 className="screen__title">What did the regulation say, and on what day?</h1>
        <p className="screen__lede">
          Retrieval is predicated on the date <em>and</em> on who is asking, so the answer is
          built only from text that was in force and that governs the profile below. The
          evidence appears first because it is ready first.
        </p>
      </div>

      <form onSubmit={submit} noValidate>
        <Field
          label="question"
          hint={`${q.trim().length}/512 characters — normalised (NFKC, homoglyphs folded) before retrieval`}
          bad={tooShort && q.length > 0 ? "a question needs at least two characters" : null}
        >
          {(id) => (
            <textarea
              id={id}
              className="textarea"
              value={q}
              maxLength={512}
              spellCheck
              onChange={(e) => setQ(e.target.value)}
              onKeyDown={(e) => {
                // Enter submits; Shift+Enter is a newline. A multi-line question is legal and
                // rare, and making the common case take two keys to serve the rare one is
                // the wrong trade.
                if (e.key === "Enter" && !e.shiftKey) submit();
              }}
              placeholder="Ask about 5 CFR — leave, pay, appointment, reduction in force…"
            />
          )}
        </Field>

        <div className="row" style={{ marginBottom: "0.4rem" }}>
          <span className="label" style={{ paddingBottom: "0.35rem" }}>
            try
          </span>
          {EXAMPLES.map((ex) => (
            <button
              type="button"
              key={ex}
              className="btn btn--ghost btn--sm"
              style={{ textTransform: "none", letterSpacing: 0 }}
              onClick={() => setQ(ex)}
            >
              {ex.length > 46 ? ex.slice(0, 44) + "…" : ex}
            </button>
          ))}
        </div>

        <div className="cols--3 cols" style={{ marginTop: "1.25rem", alignItems: "start" }}>
          <Field
            label="as of"
            bad={dateProblem ?? (outOfRange ? "outside the answerable range" : null)}
            hint={
              meta.earliest && meta.latest
                ? `answerable ${meta.earliest} → ${meta.latest}`
                : "the corpus reports no date range"
            }
            aside={
              <span className="row" style={{ gap: "0.3rem" }}>
                {meta.earliest ? (
                  <button
                    type="button"
                    className="btn btn--ghost btn--sm"
                    onClick={() => setAsOf(meta.earliest!)}
                  >
                    first
                  </button>
                ) : null}
                {meta.latest ? (
                  <button
                    type="button"
                    className="btn btn--ghost btn--sm"
                    onClick={() => setAsOf(meta.latest!)}
                  >
                    latest
                  </button>
                ) : null}
              </span>
            }
          >
            {(id) => (
              <input
                id={id}
                className="input"
                type="date"
                value={asOf}
                min={meta.earliest ?? undefined}
                max={meta.latest ?? undefined}
                aria-invalid={dateProblem ? true : undefined}
                onChange={(e) => setAsOf(e.target.value)}
              />
            )}
          </Field>

          {payValues.length ? (
            <Chips
              label="pay system"
              values={payValues}
              value={pay}
              onChange={setPay}
              anyLabel="any"
            />
          ) : null}
          {serviceValues.length ? (
            <Chips
              label="service"
              values={serviceValues}
              value={service}
              onChange={setService}
              anyLabel="any"
            />
          ) : null}
        </div>

        <div className="row" style={{ marginTop: "0.5rem", alignItems: "center" }}>
          <button className="btn" type="submit" disabled={!!dateProblem || tooShort || running || blocked}>
            {running ? "asking…" : "ask"}
          </button>
          {running ? (
            <button className="btn btn--ghost" type="button" onClick={cancel}>
              cancel
            </button>
          ) : state.phase !== "idle" ? (
            <button className="btn btn--ghost" type="button" onClick={reset}>
              clear
            </button>
          ) : null}
          <span className="hint" style={{ marginTop: 0 }}>
            {blocked
              ? "no corpus — nothing to ask"
              : pay || service
                ? `scope: ${[pay && `pay_system=${pay}`, service && `service=${service}`]
                    .filter(Boolean)
                    .join(", ")}`
                : "scope: government-wide"}
          </span>
        </div>
      </form>

      <Result state={state} asOf={asOf} onRetry={submit} />
    </div>
  );
}

// -- the streamed result ------------------------------------------------------------------

function Result({
  state,
  asOf,
  onRetry,
}: {
  state: AskState;
  asOf: string;
  onRetry: () => void;
}) {
  const running =
    state.phase === "opening" || state.phase === "retrieved" || state.phase === "generating";
  const now = useNow(running);

  const cited = useMemo(() => {
    const s = new Set<string>();
    for (const c of state.claims) for (const v of c.citations) s.add(v);
    return s;
  }, [state.claims]);

  if (state.phase === "idle") {
    return (
      <>
        <SectionLabel n="—">nothing asked yet</SectionLabel>
        <Empty title="The ledger fills from the top.">
          <p style={{ marginTop: "0.5rem", maxWidth: "42rem" }}>
            Retrieval lands in tens of milliseconds and its result is shown immediately.
            Generation is serialised at one at a time and takes seconds; the claims appear
            below the evidence when they have been parsed and every citation checked against
            what was actually retrieved and actually in force.
          </p>
        </Empty>
      </>
    );
  }

  const openedAt = state.openedAt ?? 0;
  const evidenceMs = state.evidenceAt !== null ? state.evidenceAt - openedAt : null;
  const finishedMs = state.finishedAt !== null ? state.finishedAt - openedAt : null;
  const liveMs = finishedMs ?? (running ? now - openedAt : null);

  return (
    <>
      {/* -- arrival ledger: the asymmetry, stated as two numbers ------------------------ */}
      <SectionLabel n="01">arrival</SectionLabel>
      <div className="panel">
        <div className="panel__body">
          <div className="kv" style={{ borderTop: 0 }}>
            <div className="kv__k">retrieval</div>
            <div className="kv__v">
              {state.retrieval ? (
                <>
                  {ms(state.retrieval.timings.total ?? 0)} server ·{" "}
                  {evidenceMs !== null ? ms(evidenceMs) : "—"} to first paint ·{" "}
                  {num(state.retrieval.admitted)} chunks admitted
                </>
              ) : (
                <span className="dim">opening the stream…</span>
              )}
            </div>
            <div className="kv__k">generation</div>
            <div className="kv__v">
              <GenerationLine state={state} liveMs={liveMs} />
            </div>
            {state.retrieval?.excluded_parts.length ? (
              <>
                <div className="kv__k">excluded</div>
                <div className="kv__v">
                  {state.retrieval.excluded_parts.map((p) => (
                    <Tag key={p}>part {p}</Tag>
                  ))}{" "}
                  <span className="dim">— the scope does not govern these</span>
                </div>
              </>
            ) : null}
            {state.done?.trace_id ? (
              <>
                <div className="kv__k">trace</div>
                <div className="kv__v">
                  <Copy value={state.done.trace_id} />
                </div>
              </>
            ) : null}
          </div>
          {state.retrieval ? <Timings timings={state.retrieval.timings} /> : null}
        </div>
      </div>

      {/* -- refusal --------------------------------------------------------------------- */}
      {state.error ? (
        <div style={{ marginTop: "1rem" }}>
          <Refusal
            status={state.error.status}
            detail={state.error.detail}
            retryAfter={state.error.retryAfter}
            onRetry={onRetry}
          />
          {state.evidence.length ? (
            <p className="hint" style={{ marginTop: "0.6rem" }}>
              The refusal replaced the <span className="mono">done</span> frame. Everything
              already delivered is still below — evidence that arrived is still evidence.
            </p>
          ) : null}
        </div>
      ) : null}

      {state.phase === "cancelled" ? (
        <div style={{ marginTop: "1rem" }}>
          <Note
            kind="warn"
            label="cancelled"
            title="You stopped the stream."
            detail="The socket is closed, but a generation already inside the slot runs to completion — the server has one, and it is not interruptible."
          />
        </div>
      ) : null}

      {/* -- evidence -------------------------------------------------------------------- */}
      <SectionLabel n="02">
        evidence{state.evidence.length ? ` — ${state.evidence.length} chunks` : ""}
      </SectionLabel>
      {state.evidence.length === 0 ? (
        running ? (
          <div className="evidence">
            <div className="ev" style={{ paddingLeft: "1rem" }}>
              <span className="ev__rank">·</span>
              <div className="dim">retrieving…</div>
            </div>
          </div>
        ) : (
          <Empty title="Nothing was in force for that question on that date.">
            <p style={{ marginTop: "0.5rem", maxWidth: "42rem" }}>
              An empty result is a real answer here, not a failure. The predicates ran and
              admitted{" "}
              <span className="mono">{num(state.retrieval?.admitted ?? 0)}</span> chunks; none
              of them ranked. Try a wider scope, or a date inside the answerable range.
            </p>
          </Empty>
        )
      ) : (
        <div className="evidence">
          {state.evidence.map((ev, i) => (
            <EvidenceRow
              key={ev.version_id}
              ev={ev}
              rank={i + 1}
              asOf={asOf}
              cited={cited.has(ev.version_id)}
            />
          ))}
        </div>
      )}

      {/* -- claims ---------------------------------------------------------------------- */}
      <SectionLabel n="03">answer</SectionLabel>
      <Claims state={state} />
    </>
  );
}

function GenerationLine({ state, liveMs }: { state: AskState; liveMs: number | null }) {
  if (state.done && !state.done.generated) {
    return (
      <span className="dim">
        not run — the server was started with <span className="mono">--no-generate</span>, so
        this is retrieval only.
      </span>
    );
  }
  if (state.phase === "generating") {
    const elapsed = liveMs !== null && state.evidenceAt !== null ? liveMs : null;
    return (
      <>
        <span className="dim">
          in the slot · one at a time · a measured 29.2 tok/s over ~205 tokens
        </span>{" "}
        {elapsed !== null ? <strong>{(elapsed / 1000).toFixed(1)} s</strong> : null}
      </>
    );
  }
  if (state.phase === "done" && state.done) {
    return (
      <>
        {liveMs !== null ? `${(liveMs / 1000).toFixed(2)} s total` : "complete"} ·{" "}
        {state.done.abstained ? (
          <Tag kind="warn">abstained</Tag>
        ) : (
          <Tag kind="ok">answered</Tag>
        )}{" "}
        {state.done.parse_failed ? <Tag kind="stamp">parse failed</Tag> : null}
      </>
    );
  }
  if (state.phase === "refused") return <span className="dim">refused before it finished</span>;
  if (state.phase === "cancelled") return <span className="dim">cancelled</span>;
  return <span className="dim">not started</span>;
}

/** Per-stage wall clock off the `retrieval` frame, drawn against the slowest stage. */
function Timings({ timings }: { timings: Record<string, number> }) {
  const rows = Object.entries(timings).filter(([k]) => k !== "total");
  if (!rows.length) return null;
  const max = Math.max(...rows.map(([, v]) => v), 0.001);
  return (
    <div className="scroller" style={{ marginTop: "0.9rem" }}>
      <div className="meter" style={{ minWidth: "22rem" }}>
        {rows.map(([k, v]) => (
          <div key={k} style={{ display: "contents" }}>
            <span className="meter__k">{k}</span>
            <span className="meter__track">
              <span className="meter__fill" style={{ width: `${Math.max(1.5, (v / max) * 100)}%` }} />
            </span>
            <span className="meter__v">{ms(v)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function EvidenceRow({
  ev,
  rank,
  asOf,
  cited,
}: {
  ev: StreamEvidence;
  rank: number;
  asOf: string;
  cited: boolean;
}) {
  const [open, setOpen] = useState(false);
  const long = ev.text.length > 640;
  const section = sectionOf(ev.chunk_id);
  const stale = superseded(ev);
  const force = inForce(ev, asOf);
  return (
    <article
      className={`ev stamped${cited ? " ev--cited" : ""}`}
      style={{ animationDelay: `${Math.min(rank, 10) * 22}ms` }}
    >
      {stale ? <Stamp corner /> : null}
      <span className="ev__rank">{String(rank).padStart(2, "0")}</span>
      <div className="stamped__text" style={stale ? undefined : { opacity: 1, filter: "none" }}>
        <div className="ev__meta">
          <a className="ev__id" href={`#/timeline/${encodeURIComponent(section)}/${asOf}`}>
            § {ev.chunk_id}
          </a>
          {ev.heading ? <span className="ev__heading">{ev.heading}</span> : null}
          {cited ? <Tag kind="accent">cited</Tag> : null}
        </div>
        <div className="ev__text">
          {long && !open ? ev.text.slice(0, 620).trimEnd() + "…" : ev.text}
        </div>
        {long ? (
          <button className="fold" style={{ marginTop: "0.5rem" }} onClick={() => setOpen(!open)}>
            {open ? "▲ fold" : `▼ unfold — ${num(ev.text.length)} characters`}
          </button>
        ) : null}
        <div className="ev__foot">
          <span>
            {longDate(ev.valid_from)} → {ev.valid_to ? longDate(ev.valid_to) : "in force"}
          </span>
          <span className="dim">{span(ev.valid_from, ev.valid_to)}</span>
          <span className="dim">
            {AUTHORITY[ev.authority] ?? ev.source} · tier {ev.authority}
          </span>
          {force ? <Tag kind="ok">in force on {asOf}</Tag> : null}
          <Copy value={ev.version_id} label="version id" />
        </div>
      </div>
    </article>
  );
}

function Claims({ state }: { state: AskState }) {
  if (state.done && !state.done.generated) {
    return (
      <Note
        kind="plain"
        label="no prose"
        title="Generation is off on this server, so there is nothing to write an answer with."
        detail="Everything above is the retrieval half, and it is the half that is checkable."
      >
        <pre className="cmd">python -m warrant.cli serve # without --no-generate</pre>
      </Note>
    );
  }
  if (state.claims.length === 0) {
    if (state.phase === "generating") {
      return (
        <Note
          kind="accent"
          label="generating"
          title="Nothing is shown until it is real."
          detail="Tokens are deliberately not streamed: the model emits a JSON envelope of claims and evidence ids, and its partial states are half-written citations, not partial answers."
        />
      );
    }
    if (state.done?.abstained) {
      return (
        <Note
          kind="ok"
          label="abstained"
          title="The generator declined to answer from this evidence."
          detail="An abstention is a valid answer by construction — no claims, nothing to cite, nothing to check. Refusing to abstain is what produces confident wrong answers."
        />
      );
    }
    if (state.phase === "done") {
      return <Empty title="The answer came back with no claims." />;
    }
    return <Empty title="Waiting on the generator." />;
  }
  return (
    <div>
      {state.claims.map((c, i) => (
        <div
          className={`claim${c.grounded ? "" : " claim--ungrounded"}`}
          key={i}
          style={{ animationDelay: `${i * 45}ms` }}
        >
          <span className="claim__mark" aria-hidden="true">
            {c.grounded ? "▪" : "△"}
          </span>
          <div>
            <p className="claim__text">{c.text}</p>
            <div className="claim__cites">
              <span className="label" style={{ alignSelf: "center", marginRight: "0.2rem" }}>
                {c.grounded ? "grounded in" : "cites"}
              </span>
              {c.citations.map((v) => (
                <a
                  className="cite"
                  key={v}
                  href={`#/timeline/${encodeURIComponent(sectionOf(v))}/${v.split("@")[1] ?? ""}`}
                >
                  {v}
                </a>
              ))}
            </div>
            {!c.grounded ? (
              <p className="hint hint--bad">
                Not span-aligned to its cited text. The citation was retrieved and is in force —
                that is checked before this is shown — but no verbatim span backs the sentence.
              </p>
            ) : null}
          </div>
        </div>
      ))}
    </div>
  );
}
