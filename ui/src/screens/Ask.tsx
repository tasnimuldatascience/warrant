import { useEffect, useMemo, useState } from "react";
import { useCorpus } from "../app";
import { useAsk, type AskState } from "../ask";
import { api, type AskParams, type SectionVersion, type StreamEvidence } from "../api";
import {
  AUTHORITY,
  addDays,
  badDate,
  daysBetween,
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
  useAsync,
  useNow,
} from "../lib";
import {
  Chips,
  Copy,
  Empty,
  Failure,
  Field,
  Meter,
  Note,
  RegText,
  Refusal,
  SectionLabel,
  Stamp,
  Tag,
} from "../ui";

/**
 * The flagship temporal example. One question, one paragraph, two dates either side of the
 * one amendment that changed the words themselves -- not just a cross-reference, an actual
 * change of substance a reader would act on.
 *
 * §575.102's definition of "service agreement" required a term of *not less than 6 months*
 * from 2017-01-01 until it was amended on 2026-08-25 (the newest amendment in this corpus, as
 * of the date this shipped) to drop that floor entirely -- only the 4-year ceiling remains.
 * §630.306, the other obvious candidate, was checked and rejected for this role: its 2020
 * amendment only added a cross-reference, and the paragraph a visitor would actually read
 * barely moves. This one visibly changes.
 */
const HERO_SECTION = "575.102";
const HERO_ANCHOR = "p21";
const HERO_Q = "How long can a service agreement for a recruitment incentive run?";
const HERO_BEFORE = "2020-06-01";
const HERO_AFTER = "2026-08-25";

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

/**
 * One demonstration each of the three claims this project actually measures: the as-of
 * predicate changes the answer (temporal), the scope predicate changes which part governs
 * (scope), and evidence can carry its own exception clause that a generator can drop
 * (exception -- 15.7% of chunks contain except/unless/subject to). Each fills the form and
 * runs the real streamed request; nothing here is staged.
 */
interface Demo {
  kind: string;
  q: string;
  as_of?: string;
  pay_system?: string | null;
  note: string;
}

const DEMOS: Demo[] = [
  {
    kind: "temporal · before",
    q: HERO_Q,
    as_of: HERO_BEFORE,
    note: "§575.102 as it read for nine years — a service agreement had to run at least 6 months, and at most 4 years.",
  },
  {
    kind: "temporal · after",
    q: HERO_Q,
    as_of: HERO_AFTER,
    note: "The same section, the same question, asked the day of its amendment — the 6-month floor is gone; only the 4-year ceiling remains.",
  },
  {
    kind: "scope · GS",
    q: "How is my within-grade increase determined?",
    pay_system: "GS",
    note: "General Schedule pay retrieves part 531 — the rest of the corpus never costs a candidate slot.",
  },
  {
    kind: "scope · FWS",
    q: "How is my within-grade increase determined?",
    pay_system: "FWS",
    note: "The identical question under the Federal Wage System retrieves part 532 instead. Same words, different governing part.",
  },
  {
    kind: "exception",
    q: "Is there a time limit on reinstatement eligibility after career tenure?",
    note: "The top evidence reads “no time limit… except as provided in paragraph (c)” — watch whether the exception survives into the answer below.",
  },
];

/**
 * The thesis, live: the same section scrubbed across the one date that changes its answer.
 *
 * Fetches §630.306's whole version history once — the same call the Timeline screen makes —
 * and reads the in-force paragraph client-side as the control moves, rather than round-
 * tripping retrieval on every drag. That is what makes it a control a visitor will actually
 * move: no network wait between the gesture and the text changing under it.
 */
function TemporalHero({ onRun }: { onRun: (asOf: string) => void }) {
  const [section] = useAsync((s) => api.section(HERO_SECTION, s), []);
  const [asOf, setAsOf] = useState(HERO_BEFORE);

  if (section.state === "failed") {
    return (
      <section className="hero">
        <Failure error={section.error} />
      </section>
    );
  }
  if (section.state !== "ok") {
    return (
      <section className="hero hero--loading" aria-hidden="true">
        <div className="bone" style={{ width: "40%", height: "1.2rem" }} />
        <div className="bone" style={{ width: "78%", height: "3rem", marginTop: "0.9rem" }} />
      </section>
    );
  }

  const versions: SectionVersion[] = section.value.versions;
  const lo = versions[0]?.valid_from ?? HERO_BEFORE;
  const last = versions[versions.length - 1];
  const hi = last?.valid_to ?? today();
  const total = Math.max(1, daysBetween(lo, hi));
  const offset = Math.min(total, Math.max(0, daysBetween(lo, asOf)));
  const activeIdx = versions.findIndex((v) => inForce(v, asOf));
  const active = activeIdx === -1 ? last : versions[activeIdx];
  const para = active?.paragraphs.find((p) => p.anchor === HERO_ANCHOR) ?? active?.paragraphs[0];
  // The only boundary in this section's life -- everywhere else the slider just moves within
  // one answer, which is also worth feeling.
  const boundary = versions.length > 1 ? versions[1].valid_from : null;
  const boundaryPct = boundary ? (daysBetween(lo, boundary) / total) * 100 : null;
  const stale = !!active && active.valid_to !== null && active.valid_to <= today();

  return (
    <section className="hero">
      <p className="hero__eyebrow label">
        the same question, {versions.length} versions of one section — one demonstration
      </p>
      <h1 className="hero__q">{HERO_Q}</h1>

      <div className="hero__stage stamped" key={active?.valid_from}>
        {stale ? <Stamp corner /> : null}
        <div className="stamped__text" style={stale ? undefined : { opacity: 1, filter: "none" }}>
          {para ? <RegText text={para.text} className="hero__answer" /> : null}
        </div>
        <div className="hero__meta">
          <span className="mono">
            {active ? longDate(active.valid_from) : "—"} →{" "}
            {active?.valid_to ? longDate(active.valid_to) : "in force"}
          </span>
          {stale ? <Tag kind="stamp">superseded</Tag> : <Tag kind="ok">current</Tag>}
          <span className="dim">§ {HERO_SECTION}, part 630</span>
        </div>
      </div>

      <div className="hero__control">
        <label className="visually-hidden" htmlFor="hero-scrub">
          As-of date for § 630.306
        </label>
        <input
          id="hero-scrub"
          type="range"
          className="hero__slider"
          min={0}
          max={total}
          step={1}
          value={offset}
          onChange={(e) => setAsOf(addDays(lo, Number(e.target.value)))}
          aria-valuetext={longDate(asOf)}
        />
        {boundaryPct !== null ? (
          <div className="hero__ticks" aria-hidden="true">
            <span className="hero__tick" style={{ left: `${boundaryPct}%` }} />
            {/* The tick mark sits at the true position; the label clamps inside the track
                and switches which edge it hangs from near either end, so a boundary close
                to "today" -- which this corpus's newest amendment always is -- never runs
                text off the side of the control. */}
            <span
              className="hero__tick-label"
              style={
                boundaryPct > 80
                  ? {
                      right: `${Math.max(0, 100 - boundaryPct)}%`,
                      translate: "0 0",
                      textAlign: "right",
                    }
                  : boundaryPct < 20
                    ? { left: `${boundaryPct}%`, translate: "0 0", textAlign: "left" }
                    : { left: `${boundaryPct}%`, translate: "-50% 0", textAlign: "center" }
              }
            >
              amended {boundary}
            </span>
          </div>
        ) : null}
        <div className="row" style={{ marginTop: "0.5rem", alignItems: "center" }}>
          <button type="button" className="btn btn--ghost btn--sm" onClick={() => setAsOf(HERO_BEFORE)}>
            {HERO_BEFORE} · before
          </button>
          <button type="button" className="btn btn--ghost btn--sm" onClick={() => setAsOf(HERO_AFTER)}>
            {HERO_AFTER} · after
          </button>
          <span className="hint" style={{ marginTop: 0 }}>
            asked as of <strong className="mono">{asOf}</strong>
          </span>
        </div>
      </div>

      <button type="button" className="btn" onClick={() => onRun(asOf)}>
        ask this, live, as of {asOf} →
      </button>
    </section>
  );
}

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

  /** Fill the form to match a demonstration and run it for real — no staged transcript. */
  function runDemo(d: Demo) {
    if (blocked) return;
    const params: AskParams = {
      q: d.q,
      as_of: clampDate(d.as_of ?? meta.latest ?? today()),
      pay_system: d.pay_system ?? null,
      service: null,
    };
    setQ(params.q);
    setAsOf(params.as_of);
    setPay(params.pay_system ?? null);
    setService(null);
    run(params);
  }

  const payValues = meta.facets.pay_system ?? [];
  const serviceValues = meta.facets.service ?? [];

  return (
    <div className="screen">
      <p className="orient">
        Warrant answers <strong>as of a date you choose</strong>, cites every claim to a
        specific evidence id rather than to its own prose, and — when a request fails —
        names the exact pipeline stage responsible instead of a generic error. The panel
        below is the first claim, live; the third is one click away on{" "}
        <a href="#/trace">Trace</a>.
      </p>

      <TemporalHero onRun={(d) => runDemo({ kind: "temporal", q: HERO_Q, as_of: d, note: "" })} />

      <p className="demos__label label">three more things this corpus can show you</p>
      <div className="demos">
        {DEMOS.map((d, i) => (
          <button
            type="button"
            key={i}
            className="demo"
            onClick={() => runDemo(d)}
            disabled={blocked}
          >
            <span className="demo__kind label">{d.kind}</span>
            <span className="demo__q">{d.q}</span>
            <span className="demo__note">{d.note}</span>
          </button>
        ))}
      </div>

      <p className="demos__label label" style={{ marginTop: "2rem" }}>
        or ask anything else
      </p>
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
            other examples
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
          <Race state={state} evidenceMs={evidenceMs} liveMs={liveMs} />
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

/**
 * The asymmetry, drawn rather than stated: evidence fills solid the instant it arrives, and
 * prose hatches — deliberately unfinished-looking, because there is no completion fraction
 * to report honestly while a token budget is still being spent — until it either lands or
 * doesn't.
 */
function Race({
  state,
  evidenceMs,
  liveMs,
}: {
  state: AskState;
  evidenceMs: number | null;
  liveMs: number | null;
}) {
  if (state.done && state.done.generated === false) return null;
  const evDone = evidenceMs !== null;
  let proseDisplay: string;
  let proseValue = 0;
  let busy = false;
  if (state.phase === "generating") {
    busy = true;
    proseDisplay = "composing — one at a time, ~205 tokens";
  } else if (state.phase === "done" && state.done?.generated) {
    proseValue = 1;
    proseDisplay = liveMs !== null ? `${ms(liveMs)} — landed` : "landed";
  } else if (state.phase === "refused" || state.phase === "cancelled") {
    proseDisplay = "did not land";
  } else {
    proseDisplay = "not yet in the slot";
  }
  return (
    <div className="meter race" style={{ marginTop: "0.9rem" }}>
      <Meter label="evidence" value={evDone ? 1 : 0} max={1}
             display={evDone ? `${ms(evidenceMs!)} — landed` : "in flight…"} />
      <Meter label="prose" value={proseValue} max={1} display={proseDisplay} busy={busy} alt />
    </div>
  );
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
        <RegText
          text={long && !open ? ev.text.slice(0, 620).trimEnd() + "…" : ev.text}
          className="ev__text"
        />
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
