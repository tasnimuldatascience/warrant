import { api } from "../api";
import { useCorpus } from "../app";
import { useAsk } from "../ask";
import { ms, num, pct, useAsync } from "../lib";
import type { Screen } from "../lib";
import { Bones, Copy, Empty, Failure, Note, SectionLabel, Tag } from "../ui";

/**
 * Trace.
 *
 * The claim this project makes is that a wrong answer is attributable to a *stage*, so this
 * screen is that claim rendered rather than a JSON dump of the trace. Three parts, in the
 * order the argument runs:
 *
 *   01  the funnel      13,212 chunks became 8, and every narrowing is named and counted
 *   02  the stages off  four of them, each with the measurement that switched it off
 *   03  the budget      of the failures that did happen, which stage owned them
 *
 * The counts come from `AskResponse.trace` and the timings from the stream's `retrieval`
 * frame. They are two halves of one request on purpose: the stream carries what each stage
 * *cost* and the trace model carries what each stage *returned*, and neither alone answers
 * "where did this go wrong".
 */

/** Stages that ship off, with the number and the p-value that put them there. */
const OFF: { name: string; effect: string; p: string; note: string }[] = [
  {
    name: "rerank",
    effect: "+0.5 pts",
    p: "p = 0.79",
    note: "A cross-encoder over the fused top-k. It helps by less than the noise, and costs an 87 MB model on the answering path.",
  },
  {
    name: "entailment",
    effect: "+2.3 pts",
    p: "p = 0.55",
    note: "Filtering evidence by NLI entailment against the question. The largest apparent gain of the four and still not distinguishable from zero.",
  },
  {
    name: "calibrated combiner",
    effect: "AURC +0.0019",
    p: "worse",
    note: "A learned combination of the fusion signals. Moved risk-coverage the wrong way, which is the one direction a calibration stage may not move it.",
  },
  {
    name: "multi-hop",
    effect: "−0.88 pts",
    p: "p = 0.25",
    note: "Following a dangling cross-reference and retrieving again. Never once positive at any budget or depth, so closing the reference is not the same as answering better.",
  },
];

export default function Trace({ go }: { go: (screen: Screen, ...rest: string[]) => void }) {
  const { meta } = useCorpus();
  const { state } = useAsk();
  const [budget, retryBudget] = useAsync((s) => api.budget(s), []);

  const counts = state.counts;
  const timings = state.retrieval?.timings ?? null;

  return (
    <div className="screen">
      <div className="screen__head">
        <h1 className="screen__title">Which stage is answerable for the answer.</h1>
        <p className="screen__lede">
          Every answer this system emits is the output of a named sequence of narrowings. If
          the answer is wrong, exactly one of them is where it went wrong — and the point of
          recording each stage's input, output and cost is that you can say which, instead of
          re-running the whole pipeline and hoping.
        </p>
      </div>

      <SectionLabel n="01">the funnel</SectionLabel>
      {!counts && !timings ? (
        <Empty title="No request to trace yet.">
          <p style={{ marginTop: "0.5rem", maxWidth: "42rem" }}>
            Ask something and this fills in from that request — the counts arrive from a
            retrieval-only call fired alongside the stream, so tracing costs a second 18 ms
            and no generation slot.
          </p>
          <p style={{ marginTop: "0.8rem" }}>
            <button className="btn btn--ghost btn--sm" onClick={() => go("ask")}>
              go to ask
            </button>
          </p>
        </Empty>
      ) : (
        <Funnel counts={counts} timings={timings} corpus={meta.chunks} />
      )}

      {state.countsError ? (
        <div style={{ marginTop: "1rem" }}>
          <Failure error={state.countsError} />
        </div>
      ) : null}

      {state.params || state.done?.trace_id ? (
        <div className="panel" style={{ marginTop: "1rem" }}>
          <div className="panel__head">
            <span className="label">the request</span>
            {state.done?.trace_id ? <Copy value={state.done.trace_id} label="trace id" /> : null}
          </div>
          <div className="panel__body">
            <div className="kv">
              {state.params ? (
                <>
                  <div className="kv__k">question</div>
                  <div className="kv__v" style={{ fontFamily: "var(--serif)" }}>
                    {state.params.q}
                  </div>
                  <div className="kv__k">as of</div>
                  <div className="kv__v">{state.params.as_of}</div>
                  <div className="kv__k">scope</div>
                  <div className="kv__v">
                    {state.params.pay_system || state.params.service
                      ? [
                          state.params.pay_system && `pay_system=${state.params.pay_system}`,
                          state.params.service && `service=${state.params.service}`,
                        ]
                          .filter(Boolean)
                          .join(", ")
                      : "government-wide"}
                  </div>
                </>
              ) : null}
              {state.retrieval?.excluded_parts.length ? (
                <>
                  <div className="kv__k">parts excluded</div>
                  <div className="kv__v">{state.retrieval.excluded_parts.join(", ")}</div>
                </>
              ) : null}
              <div className="kv__k">outcome</div>
              <div className="kv__v">
                {state.phase === "done" ? (
                  state.done?.generated === false ? (
                    <Tag>retrieval only</Tag>
                  ) : state.done?.abstained ? (
                    <Tag kind="warn">abstained</Tag>
                  ) : (
                    <Tag kind="ok">answered</Tag>
                  )
                ) : state.phase === "refused" ? (
                  <Tag kind="stamp">refused · {state.error?.status}</Tag>
                ) : (
                  <Tag>{state.phase}</Tag>
                )}
              </div>
              <div className="kv__k">config</div>
              <div className="kv__v">{meta.config_hash}</div>
            </div>
            {!state.done?.trace_id && state.phase === "done" ? (
              <p className="hint" style={{ marginTop: "0.7rem" }}>
                No trace id: recording is off, or the write failed. The answer is still valid —
                only the audit row is missing, and that asymmetry is deliberate. An audit trail
                that can take the service down is a liability.
              </p>
            ) : null}
          </div>
        </div>
      ) : null}

      <SectionLabel n="02">stages that did not run</SectionLabel>
      <p style={{ maxWidth: "46rem", marginBottom: "1rem", color: "var(--ink-2)" }}>
        Four stages are built, tested, and shipped <em>off</em>. Each is listed with the
        measurement that switched it off, because a stage disabled without a number beside it
        is indistinguishable from a stage nobody finished.
      </p>
      <div className="panel">
        <div className="scroller">
          <table className="ledger" style={{ minWidth: "38rem" }}>
            <thead>
              <tr>
                <th>stage</th>
                <th className="n">effect</th>
                <th className="n">significance</th>
                <th>why it is off</th>
              </tr>
            </thead>
            <tbody>
              {OFF.map((s) => (
                <tr key={s.name}>
                  <td className="mono">{s.name}</td>
                  <td className="n">{s.effect}</td>
                  <td className="n dim">{s.p}</td>
                  <td style={{ maxWidth: "26rem" }}>{s.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <SectionLabel n="03">the failure budget</SectionLabel>
      {budget.state === "loading" || budget.state === "idle" ? (
        <Bones rows={4} />
      ) : budget.state === "failed" ? (
        <Failure error={budget.error} onRetry={retryBudget} />
      ) : (
        <Budget value={budget.value} />
      )}
    </div>
  );
}

// -- funnel ---------------------------------------------------------------------------------

/** Which retrieval stages run against the same candidate pool rather than in sequence. */
const PARALLEL = new Set(["lexical", "dense"]);

function Funnel({
  counts,
  timings,
  corpus,
}: {
  counts: { admitted: number; corpus: number; stages: { name: string; out: number }[] } | null;
  timings: Record<string, number> | null;
  corpus: number;
}) {
  const stages = counts?.stages ?? [];
  const maxOut = Math.max(corpus, ...stages.map((s) => s.out), 1);
  const times = timings ?? {};
  const maxTime = Math.max(...Object.entries(times).filter(([k]) => k !== "total").map(([, v]) => v), 0.001);

  return (
    <div className="panel">
      <div className="panel__head">
        <span className="label">chunks in, chunks out, milliseconds spent</span>
        {times.total ? <span className="mono">{ms(times.total)} total</span> : null}
      </div>
      <div className="panel__body">
        <div className="scroller">
          <div className="funnel">
            <div className="funnel__head">stage</div>
            <div className="funnel__head">share of corpus</div>
            <div className="funnel__head" style={{ textAlign: "right" }}>
              out
            </div>
            <div className="funnel__head">time</div>
            <div className="funnel__head" style={{ textAlign: "right" }}>
              ms
            </div>

            <div className="funnel__row">
              <div className="funnel__name">corpus</div>
              <div className="funnel__bar">
                <div className="funnel__track">
                  <div className="funnel__fill" style={{ width: "100%" }} />
                </div>
              </div>
              <div className="funnel__count">{num(corpus)}</div>
              <div className="funnel__time" />
              <div className="funnel__tv dim">—</div>
            </div>

            {stages.map((s) => {
              const t = times[s.name];
              return (
                <div
                  className={`funnel__row${PARALLEL.has(s.name) ? " funnel__row--parallel" : ""}`}
                  key={s.name}
                >
                  <div className="funnel__name">{s.name}</div>
                  <div className="funnel__bar">
                    <div className="funnel__track">
                      <div
                        className="funnel__fill"
                        style={{ width: `${Math.max(0.4, (s.out / maxOut) * 100)}%` }}
                      />
                    </div>
                  </div>
                  <div className="funnel__count">
                    {s.out === 0 ? <span className="dim">0</span> : num(s.out)}
                  </div>
                  <div className="funnel__time">
                    <div className="funnel__track">
                      <div
                        className="funnel__fill funnel__fill--time"
                        style={{ width: t ? `${Math.max(1, (t / maxTime) * 100)}%` : "0%" }}
                      />
                    </div>
                  </div>
                  <div className="funnel__tv">{t !== undefined ? ms(t) : <span className="dim">—</span>}</div>
                </div>
              );
            })}
          </div>
        </div>

        {!counts ? (
          <p className="hint" style={{ marginTop: "0.8rem" }}>
            Timings only — the retrieval-only call that carries the per-stage counts has not
            returned yet, or was refused.
          </p>
        ) : (
          <div className="kv" style={{ marginTop: "1rem" }}>
            <div className="kv__k">predicate narrowing</div>
            <div className="kv__v">
              {num(corpus)} → {num(counts.admitted)} ·{" "}
              <span className="dim">
                {pct(1 - counts.admitted / Math.max(1, corpus))} of the corpus could not have been
                in force, or could not govern this profile, and never cost a candidate slot
              </span>
            </div>
            <div className="kv__k">stages with 0 out</div>
            <div className="kv__v">
              {stages.filter((s) => s.out === 0).length === 0 ? (
                <span className="dim">none</span>
              ) : (
                <>
                  {stages
                    .filter((s) => s.out === 0)
                    .map((s) => (
                      <Tag key={s.name}>{s.name}</Tag>
                    ))}{" "}
                  <span className="dim">
                    — a stage that ships off returns nothing; that is the flag, not a fault
                  </span>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// -- budget ----------------------------------------------------------------------------------

function Budget({
  value,
}: {
  value: {
    bucket: string;
    config_hash: string;
    items: number;
    failures: number;
    success_rate: number;
    observational: { stage: string; failures: number; share: string }[];
    interventional: { repair: string; implicated: number }[];
    stages: string[];
  };
}) {
  const owned = new Map(value.observational.map((o) => [o.stage, o]));
  return (
    <>
      <Note
        kind={value.failures === 0 ? "ok" : "accent"}
        label={pct(value.success_rate, 2)}
        title={
          <>
            {num(value.failures)} failure{value.failures === 1 ? "" : "s"} in{" "}
            {num(value.items)} items on <span className="mono">{value.bucket}</span>
          </>
        }
        detail={
          <>
            Read from a recorded run at config <span className="mono">{value.config_hash}</span>,
            never recomputed per request — a dashboard that recomputes is a dashboard that
            drifts from the numbers in results/.
          </>
        }
      />
      <div className="cols" style={{ marginTop: "1rem" }}>
        <div className="panel">
          <div className="panel__head">
            <span className="label">observational — where it broke</span>
          </div>
          <div className="panel__body" style={{ paddingTop: 0 }}>
            <table className="ledger">
              <thead>
                <tr>
                  <th>stage</th>
                  <th className="n">failures</th>
                  <th className="n">share</th>
                </tr>
              </thead>
              <tbody>
                {value.stages.map((s) => {
                  const row = owned.get(s);
                  return (
                    <tr key={s} style={row ? undefined : { color: "var(--ink-4)" }}>
                      <td className="mono">{s}</td>
                      <td className="n">{row ? num(row.failures) : "0"}</td>
                      <td className="n">{row ? <Tag kind="stamp">{row.share}</Tag> : "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="hint" style={{ marginTop: "0.7rem" }}>
              Every stage in the pipeline is listed, including the ones that owned nothing.
              A budget that lists only the stages that failed cannot show you the ones that
              did not.
            </p>
          </div>
        </div>

        <div className="panel">
          <div className="panel__head">
            <span className="label">interventional — what would fix it</span>
          </div>
          <div className="panel__body" style={{ paddingTop: 0 }}>
            <table className="ledger">
              <thead>
                <tr>
                  <th>repair</th>
                  <th className="n">items implicated</th>
                </tr>
              </thead>
              <tbody>
                {value.interventional.length === 0 ? (
                  <tr>
                    <td colSpan={2} className="dim">
                      nothing to repair
                    </td>
                  </tr>
                ) : (
                  value.interventional.map((r) => (
                    <tr key={r.repair}>
                      <td className="mono">{r.repair}</td>
                      <td className="n">{num(r.implicated)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
            <p className="hint" style={{ marginTop: "0.7rem" }}>
              A separate question from the one on the left. Where a failure was <em>observed</em>
              is not the same as what, changed, would have prevented it — and the two columns
              disagreeing is information rather than an inconsistency.
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
