import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { ms, num, pct, secs } from "../lib";
import {
  bucketQuantile,
  filterSamples,
  findHistogram,
  findSample,
  parseMetrics,
  type HistogramSeries,
  type ParsedMetrics,
} from "../metrics";
import { Bones, Empty, Failure, Meter, Note, SectionLabel, Tag } from "../ui";

/**
 * Observability.
 *
 * Trace shows one request's stages. This is the same claim -- a wrong or slow answer is
 * attributable to a stage, not a mystery -- summed across every request the process has
 * served, from the counters and histograms it already publishes at `/metrics` and that
 * nothing before this screen rendered. Six sections, in the order the README's own argument
 * runs: retrieval is the part measured against a specific latency budget, generation against
 * a specific throughput ceiling, admission control against a specific counter that has never
 * fired, and the corpus gauges last because they describe the process, not a request.
 *
 * `/metrics` is scraped, not pushed: on a 5s interval, paused when the tab is hidden, with a
 * pause control and a visible "as of" so a reader can tell live numbers from a frozen screen.
 * A failed scrape shows as a failure, never as the previous scrape presented as current.
 */

const STAGES = ["predicates", "lexical", "dense", "fusion", "rerank"] as const;

// eval-010: 29.2-29.9 tok/s over ~205 output tokens, ceiling measured under load, 6.7-9.4
// CI. A constant, not a live computation -- README's own number, cited rather than re-derived
// from a handful of scraped observations that would not carry the same weight.
const GENERATE_CEILING_PER_MIN = 7.7;
const GENERATE_CEILING_CI = "6.7–9.4";

export default function Observability() {
  const { parsed, error, lastFetched, loading, paused, setPaused, refetch } = useMetricsPoll(5000);

  const seriesCount = useMemo(
    () => (parsed ? Object.values(parsed.samples).reduce((a, s) => a + s.length, 0) : null),
    [parsed],
  );
  const totalRequests = parsed ? totalOf(parsed, "warrant_requests_total") : 0;

  return (
    <div className="screen">
      <div className="screen__head">
        <h1 className="screen__title">Every request this process has served, aggregated.</h1>
        <p className="screen__lede">
          The server hand-emits a Prometheus scrape at <code className="mono">/metrics</code>{" "}
          and nothing rendered it before this screen. Parsed here, in the browser, from the
          text exposition format itself -- no client library, matching the server's own choice
          not to carry one.
        </p>
      </div>

      <Toolbar
        paused={paused}
        onTogglePause={() => setPaused((p) => !p)}
        lastFetched={lastFetched}
        onRefresh={refetch}
        loading={loading}
        seriesCount={seriesCount}
      />

      {error ? (
        <Failure error={error} onRetry={refetch} />
      ) : !parsed ? (
        <Bones rows={6} />
      ) : (
        <>
          {parsed.errors.length > 0 ? (
            <Note
              kind="warn"
              label="partial scrape"
              title={`${parsed.errors.length} line${parsed.errors.length === 1 ? "" : "s"} in this scrape did not parse and were skipped.`}
              detail={parsed.errors.slice(0, 3).join(" · ")}
            >
              <p className="hint" style={{ marginTop: "0.5rem" }}>
                Everything below is built from what did parse. A scrape torn mid-write should
                degrade to fewer numbers, not a blank screen or a wrong one.
              </p>
            </Note>
          ) : null}

          {totalRequests === 0 ? (
            <div style={{ marginTop: parsed.errors.length > 0 ? "0.75rem" : 0 }}>
              <Note
                kind="plain"
                label="fresh process"
                title="This process has served no requests since it started."
              >
                <p className="hint" style={{ marginTop: "0.5rem" }}>
                  Every zero below reads as "has not happened yet," not "happened and came back
                  empty." Ask something and reload this screen, or leave it open -- it polls.
                </p>
              </Note>
            </div>
          ) : null}

          <SectionLabel n="01">stage latency</SectionLabel>
          <StageLatency parsed={parsed} />

          <SectionLabel n="02">requests, by endpoint and status</SectionLabel>
          <RequestVolume parsed={parsed} />

          <SectionLabel n="03">generation</SectionLabel>
          <Generation parsed={parsed} />

          <SectionLabel n="04">admission control</SectionLabel>
          <Admission parsed={parsed} />

          <SectionLabel n="05">answer cache</SectionLabel>
          <Cache parsed={parsed} />

          <SectionLabel n="06">corpus</SectionLabel>
          <CorpusGauges parsed={parsed} />
        </>
      )}
    </div>
  );
}

// -- polling -----------------------------------------------------------------------------

function useMetricsPoll(intervalMs: number) {
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [lastFetched, setLastFetched] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [paused, setPaused] = useState(false);
  const inflight = useRef<AbortController | null>(null);

  const fetchOnce = useCallback(() => {
    inflight.current?.abort();
    const ac = new AbortController();
    inflight.current = ac;
    setLoading(true);
    api.metricsText(ac.signal).then(
      (t) => {
        setText(t);
        setError(null);
        setLastFetched(Date.now());
        setLoading(false);
      },
      (err) => {
        if ((err as Error)?.name === "AbortError") return;
        // The previous scrape's numbers stay in `text`, but `error` being set is what keeps
        // the render branch from showing them as current -- see the ternary in the screen body.
        setError(err);
        setLoading(false);
      },
    );
  }, []);

  useEffect(() => {
    fetchOnce();
    return () => inflight.current?.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (paused) return;
    const tick = () => {
      if (document.hidden) return; // never poll a tab nobody is looking at
      fetchOnce();
    };
    const id = window.setInterval(tick, intervalMs);
    // A tab that comes back from hidden should not sit on a stale scrape until the next
    // interval happens to land -- catch up the moment it is visible again.
    document.addEventListener("visibilitychange", tick);
    return () => {
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", tick);
    };
  }, [paused, intervalMs, fetchOnce]);

  const parsed = useMemo(() => (text !== null ? parseMetrics(text) : null), [text]);
  return { parsed, error, lastFetched, loading, paused, setPaused, refetch: fetchOnce };
}

function Toolbar({
  paused,
  onTogglePause,
  lastFetched,
  onRefresh,
  loading,
  seriesCount,
}: {
  paused: boolean;
  onTogglePause: () => void;
  lastFetched: number | null;
  onRefresh: () => void;
  loading: boolean;
  seriesCount: number | null;
}) {
  return (
    <div className="toolbar">
      <button className="btn btn--ghost btn--sm" aria-pressed={paused} onClick={onTogglePause}>
        {paused ? "resume polling" : "pause polling"}
      </button>
      <button className="btn btn--ghost btn--sm" onClick={onRefresh} disabled={loading}>
        {loading ? "refreshing…" : "refresh now"}
      </button>
      <span className="toolbar__asof">
        as of {lastFetched ? new Date(lastFetched).toLocaleTimeString() : "—"}
        {paused ? " · polling paused" : " · every 5s"}
      </span>
      <span className="toolbar__spacer" />
      {seriesCount !== null ? (
        <span className="toolbar__asof">{num(seriesCount)} series this scrape</span>
      ) : null}
    </div>
  );
}

// -- helpers -------------------------------------------------------------------------------

function totalOf(m: ParsedMetrics, name: string, match: Record<string, string> = {}): number {
  return filterSamples(m, name, match).reduce((a, s) => a + s.value, 0);
}

/** p50/p95 rendered together, with the "≥" honesty marker for a clipped (+Inf-bucket) quantile. */
function QuantilePair({ h }: { h: HistogramSeries | null }) {
  const p50 = h ? bucketQuantile(h, 0.5) : null;
  const p95 = h ? bucketQuantile(h, 0.95) : null;
  return (
    <>
      <td className="n">
        {p50 ? (
          <>
            {p50.clipped ? "≥ " : ""}
            {ms(p50.value)}
          </>
        ) : (
          <span className="dim">—</span>
        )}
      </td>
      <td className="n">
        {p95 ? (
          <>
            {p95.clipped ? "≥ " : ""}
            {ms(p95.value)}
          </>
        ) : (
          <span className="dim">—</span>
        )}
      </td>
    </>
  );
}

// -- 01 stage latency ------------------------------------------------------------------------

function StageLatency({ parsed }: { parsed: ParsedMetrics }) {
  const rows = STAGES.map((stage) => {
    const h = findHistogram(parsed, "warrant_stage_duration_ms", { stage });
    return { stage, h, p95: h ? bucketQuantile(h, 0.95) : null };
  });
  const anyObserved = rows.some((r) => (r.h?.count ?? 0) > 0);
  const maxP95 = Math.max(0.001, ...rows.map((r) => r.p95?.value ?? 0));

  return (
    <div className="panel">
      <div className="panel__head">
        <span className="label">wall time per retrieval stage, as recorded on the trace</span>
      </div>
      <div className="panel__body">
        {!anyObserved ? (
          <Empty title="No retrieval stage has been observed yet.">
            <p style={{ marginTop: "0.5rem" }}>
              Ask something on the Ask screen -- every stage records its own wall time on the
              trace, and this is that same number, summed across every request since the
              process started.
            </p>
          </Empty>
        ) : (
          <div className="scroller">
            <table className="ledger" style={{ minWidth: "42rem" }}>
              <thead>
                <tr>
                  <th>stage</th>
                  <th className="n">observations</th>
                  <th className="n">p50</th>
                  <th className="n">p95</th>
                  <th>relative p95</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.stage}>
                    <td className="mono">{r.stage}</td>
                    <td className="n">
                      {r.h?.count ? num(r.h.count) : <span className="dim">0</span>}
                    </td>
                    <QuantilePair h={r.h} />
                    <td>
                      <div className="bar">
                        <div
                          className="bar__fill"
                          style={{
                            width: r.p95 ? `${Math.max(2, (r.p95.value / maxP95) * 100)}%` : "0%",
                          }}
                        />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="hint" style={{ marginTop: "0.8rem" }}>
          p50 and p95 are <strong>bucket-interpolated</strong>, not exact order statistics --
          each is linear interpolation within whichever of{" "}
          <code className="mono">LATENCY_BUCKETS_MS</code>'s edges the rank falls between, dense
          from 1&ndash;40ms and thin past 250ms because that is where this project's own
          measurements put the mass. A "≥" marks a quantile that fell past the widest
          finite edge: a floor, not a number that looks precise.
        </p>
      </div>
    </div>
  );
}

// -- 02 requests -----------------------------------------------------------------------------

const STATUS_ORDER = ["2xx", "3xx", "4xx", "5xx"];

function RequestVolume({ parsed }: { parsed: ParsedMetrics }) {
  const rows = filterSamples(parsed, "warrant_requests_total");
  if (rows.length === 0) {
    return <Empty title="No requests recorded yet." />;
  }

  const byEndpoint = new Map<string, Record<string, number>>();
  for (const s of rows) {
    const ep = s.labels.endpoint ?? "—";
    const st = s.labels.status ?? "—";
    const rec = byEndpoint.get(ep) ?? {};
    rec[st] = (rec[st] ?? 0) + s.value;
    byEndpoint.set(ep, rec);
  }
  const statuses = Array.from(new Set(rows.map((s) => s.labels.status ?? "—"))).sort(
    (a, b) => STATUS_ORDER.indexOf(a) - STATUS_ORDER.indexOf(b) || a.localeCompare(b),
  );
  const endpoints = Array.from(byEndpoint.keys()).sort();
  const totalAll = rows.reduce((a, s) => a + s.value, 0);

  return (
    <div className="panel">
      <div className="panel__head">
        <span className="label">endpoint, matched to its route template -- never a raw path</span>
        <span className="mono">{num(totalAll)} total</span>
      </div>
      <div className="panel__body" style={{ paddingTop: 0 }}>
        <div className="scroller">
          <table className="ledger">
            <thead>
              <tr>
                <th>endpoint</th>
                <th className="n">total</th>
                {statuses.map((s) => (
                  <th className="n" key={s}>
                    {s}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {endpoints.map((ep) => {
                const rec = byEndpoint.get(ep)!;
                const total = Object.values(rec).reduce((a, b) => a + b, 0);
                return (
                  <tr key={ep}>
                    <td className="mono">{ep}</td>
                    <td className="n">{num(total)}</td>
                    {statuses.map((s) => {
                      const v = rec[s] ?? 0;
                      const bad = s.startsWith("5");
                      return (
                        <td className="n" key={s}>
                          {v === 0 ? (
                            <span className="dim">0</span>
                          ) : bad ? (
                            <Tag kind="stamp">{num(v)}</Tag>
                          ) : (
                            num(v)
                          )}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// -- 03 generation ---------------------------------------------------------------------------

function Generation({ parsed }: { parsed: ParsedMetrics }) {
  const h = findHistogram(parsed, "warrant_generate_duration_s");
  const count = h?.count ?? 0;

  if (!h || count === 0) {
    return (
      <Empty title="No generation call has completed yet.">
        <p style={{ marginTop: "0.5rem" }}>
          Generation is the slow half of a request -- a full answer is measured at 6.6s against
          an 18.4ms retrieval -- so this section fills in only once one has finished.
        </p>
      </Empty>
    );
  }

  const p50 = bucketQuantile(h, 0.5);
  const p95 = bucketQuantile(h, 0.95);
  const meanS = h.sum !== null && h.count ? h.sum / h.count : null;
  const impliedPerMin = meanS && meanS > 0 ? 60 / meanS : null;

  return (
    <div className="panel">
      <div className="panel__head">
        <span className="label">wall time per generation call</span>
        <span className="mono">{num(count)} calls</span>
      </div>
      <div className="panel__body">
        <div className="kv">
          <div className="kv__k">p50</div>
          <div className="kv__v">
            {p50 ? `${p50.clipped ? "≥ " : ""}${secs(p50.value)}` : "—"}
          </div>
          <div className="kv__k">p95</div>
          <div className="kv__v">
            {p95 ? `${p95.clipped ? "≥ " : ""}${secs(p95.value)}` : "—"}
          </div>
          <div className="kv__k">mean</div>
          <div className="kv__v">{meanS !== null ? secs(meanS) : "—"}</div>
          <div className="kv__k">implied throughput</div>
          <div className="kv__v">
            {impliedPerMin !== null ? `${impliedPerMin.toFixed(1)} / min` : "—"}{" "}
            <span className="dim">
              against a measured ceiling of {GENERATE_CEILING_PER_MIN} / min ({GENERATE_CEILING_CI}{" "}
              CI, eval-010) -- this process's own mean, not a re-derivation of the README's number
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

// -- 04 admission control ---------------------------------------------------------------------

function Admission({ parsed }: { parsed: ParsedMetrics }) {
  const qf = totalOf(parsed, "warrant_admission_rejected_total", { reason: "queue_full" });
  const dl = totalOf(parsed, "warrant_admission_rejected_total", { reason: "deadline" });
  const generated = findHistogram(parsed, "warrant_generate_duration_s")?.count ?? 0;
  const totalDecisions = generated + qf + dl;

  return (
    <div className="panel">
      <div className="panel__head">
        <span className="label">refused by admission control, by reason</span>
      </div>
      <div className="panel__body">
        <div className="kv">
          <div className="kv__k">queue_full</div>
          <div className="kv__v">
            {qf === 0 ? <span className="dim">0</span> : <Tag kind="warn">{num(qf)}</Tag>} of{" "}
            {num(totalDecisions)} generation attempts
          </div>
          <div className="kv__k">deadline</div>
          <div className="kv__v">
            {dl === 0 ? <span className="dim">0</span> : <Tag kind="stamp">{num(dl)}</Tag>} of{" "}
            {num(totalDecisions)} generation attempts
          </div>
        </div>
        {dl === 0 ? (
          <p className="hint" style={{ marginTop: "0.7rem" }}>
            deadline has fired zero times
            {totalDecisions > 0
              ? ` across ${num(totalDecisions)} generation attempts this process has made`
              : " -- no generation attempt has been made yet"}
            . A counter that is always zero is a claim worth showing rather than hiding: it
            means the queue wait has never once out-run the request deadline, on this traffic,
            so far -- not that it can't.
          </p>
        ) : null}
      </div>
    </div>
  );
}

// -- 05 cache ---------------------------------------------------------------------------------

function Cache({ parsed }: { parsed: ParsedMetrics }) {
  const hit = totalOf(parsed, "warrant_cache_total", { outcome: "hit" });
  const miss = totalOf(parsed, "warrant_cache_total", { outcome: "miss" });
  const total = hit + miss;

  if (total === 0) {
    return <Empty title="No cache lookup recorded yet." />;
  }

  return (
    <div className="panel">
      <div className="panel__head">
        <span className="label">answer cache, by outcome</span>
        <span className="mono">{pct(hit / total)} hit rate</span>
      </div>
      <div className="panel__body">
        <div className="meter">
          <Meter label="hit" value={hit} max={total} display={`${num(hit)} · ${pct(hit / total)}`} />
          <Meter
            label="miss"
            value={miss}
            max={total}
            display={`${num(miss)} · ${pct(miss / total)}`}
            alt
          />
        </div>
      </div>
    </div>
  );
}

// -- 06 corpus --------------------------------------------------------------------------------

function CorpusGauges({ parsed }: { parsed: ParsedMetrics }) {
  const chunks = findSample(parsed, "warrant_corpus_chunks");
  const uncovered = findSample(parsed, "warrant_uncovered_chunks");
  const ready = findSample(parsed, "warrant_ready");

  return (
    <div className="panel">
      <div className="panel__head">
        <span className="label">sampled fresh at scrape time, not tracked continuously</span>
      </div>
      <div className="panel__body">
        <div className="kv">
          <div className="kv__k">chunks</div>
          <div className="kv__v">{chunks !== null ? num(chunks) : <span className="dim">—</span>}</div>
          <div className="kv__k">uncovered</div>
          <div className="kv__v">
            {uncovered === null ? (
              <>
                <span className="dim">null</span>{" "}
                <span className="dim">
                  -- no dense index is loaded; there is nothing to be missing a vector from,
                  which is a different claim from zero missing
                </span>
              </>
            ) : uncovered === 0 ? (
              <>
                0 <span className="dim">-- every believed chunk has a vector</span>
              </>
            ) : (
              <Tag kind="warn">{num(uncovered)}</Tag>
            )}
          </div>
          <div className="kv__k">ready</div>
          <div className="kv__v">
            {ready === null ? (
              <span className="dim">—</span>
            ) : ready === 1 ? (
              <Tag kind="ok">1 · answering</Tag>
            ) : (
              <Tag kind="stamp">0 · not answering</Tag>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
