/**
 * A hand-rolled parser for the Prometheus text exposition format, matching what
 * `src/warrant/serve/metrics.py`'s `Registry.render()` writes: `# HELP`/`# TYPE` comments,
 * one line per counter/gauge sample, and four lines per histogram bucket edge (`_bucket`,
 * repeated per `le`) plus one `_sum` and one `_count`.
 *
 * This is deliberately not a regex-per-line parser. Label values are allowed to contain
 * commas, braces and escaped quotes (`_escape` in metrics.py backslash-escapes `\`, `"` and
 * `\n`), so splitting a label block on `,` before finding the real end of it is wrong on
 * exactly the inputs a naive parser would never see in its own tests -- a route template
 * containing `,` would be unlikely today but is not disallowed by the format. Nothing here
 * throws: a scrape that fails to sample a gauge, or a line truncated mid-write by a client
 * that read the response before the server finished, is still a response worth rendering as
 * much of as parses, with the rest reported rather than silently dropped.
 */

// -- types -----------------------------------------------------------------------------

export interface Sample {
  labels: Record<string, string>;
  value: number;
}

export type MetricType = "counter" | "gauge" | "histogram" | "summary" | "untyped";

export interface HistogramBucket {
  /** Upper bound, inclusive. `Infinity` for the `le="+Inf"` row. */
  le: number;
  /** Cumulative count at this bound -- already cumulative on the wire, per Registry.render(). */
  cumulative: number;
}

export interface HistogramSeries {
  /** Labels shared by this series' buckets/_sum/_count -- `le` stripped out. */
  labels: Record<string, string>;
  /** Ascending by `le`; the last entry is `+Inf` when the scrape carried one. */
  buckets: HistogramBucket[];
  /** `null` when `_sum` was not present for this label set -- a partial scrape, not a zero. */
  sum: number | null;
  /** `null` when `_count` was not present. */
  count: number | null;
}

export interface ParsedMetrics {
  help: Record<string, string>;
  type: Record<string, MetricType>;
  /** Plain (non-histogram) samples, keyed by the exact metric name on the wire. */
  samples: Record<string, Sample[]>;
  /** Histogram series, keyed by the base name (`_bucket`/`_sum`/`_count` stripped). */
  histograms: Record<string, HistogramSeries[]>;
  /** Lines that did not parse as a HELP/TYPE comment or a sample, verbatim and capped. */
  errors: string[];
}

// -- label block scanning ---------------------------------------------------------------

/** Find the `}` closing the label block opened at `line[openIdx]`, skipping quoted content. */
function findLabelBlockEnd(line: string, openIdx: number): number {
  let inQuotes = false;
  for (let i = openIdx + 1; i < line.length; i++) {
    const c = line[i];
    if (c === "\\" && inQuotes) {
      i++; // an escaped character inside a value can't close the block or the quote
      continue;
    }
    if (c === '"') inQuotes = !inQuotes;
    else if (c === "}" && !inQuotes) return i;
  }
  return -1;
}

function unescapeLabelValue(raw: string): string {
  let out = "";
  for (let i = 0; i < raw.length; i++) {
    const c = raw[i];
    if (c === "\\" && i + 1 < raw.length) {
      const n = raw[i + 1];
      if (n === "\\" || n === '"') {
        out += n;
        i++;
        continue;
      }
      if (n === "n") {
        out += "\n";
        i++;
        continue;
      }
      // An escape sequence this format never emits. Keep the backslash literally rather than
      // eating the next character on a guess -- an unrecognised escape is data, not noise.
    }
    out += c;
  }
  return out;
}

const LABEL_KEY = /[a-zA-Z0-9_]/;

function parseLabelBlock(block: string): Record<string, string> | null {
  const labels: Record<string, string> = {};
  let i = 0;
  const n = block.length;
  while (i < n) {
    while (i < n && (block[i] === " " || block[i] === ",")) i++;
    if (i >= n) break;
    const keyStart = i;
    while (i < n && LABEL_KEY.test(block[i])) i++;
    if (i === keyStart) return null;
    const key = block.slice(keyStart, i);
    while (i < n && block[i] === " ") i++;
    if (block[i] !== "=") return null;
    i++;
    while (i < n && block[i] === " ") i++;
    if (block[i] !== '"') return null;
    i++;
    const valStart = i;
    while (i < n && block[i] !== '"') {
      if (block[i] === "\\") i++; // skip the escaped character too, whatever it is
      i++;
    }
    if (i >= n) return null; // unterminated quote
    labels[key] = unescapeLabelValue(block.slice(valStart, i));
    i++; // closing quote
    while (i < n && block[i] === " ") i++;
    if (i < n && block[i] === ",") i++;
    else if (i < n) return null; // junk between fields
  }
  return labels;
}

function parseValueToken(tok: string): number | null {
  if (tok === "+Inf" || tok === "Inf") return Infinity;
  if (tok === "-Inf") return -Infinity;
  if (tok === "NaN") return NaN;
  if (tok === "") return null;
  const v = Number(tok);
  return Number.isNaN(v) ? null : v;
}

const NAME_START = /[a-zA-Z_:]/;
const NAME_REST = /[a-zA-Z0-9_:]/;

function parseSampleLine(line: string): { name: string; labels: Record<string, string>; value: number } | null {
  let i = 0;
  const n = line.length;
  if (i >= n || !NAME_START.test(line[i])) return null;
  const nameStart = i;
  i++;
  while (i < n && NAME_REST.test(line[i])) i++;
  const name = line.slice(nameStart, i);

  let labels: Record<string, string> = {};
  if (line[i] === "{") {
    const close = findLabelBlockEnd(line, i);
    if (close === -1) return null;
    const parsed = parseLabelBlock(line.slice(i + 1, close));
    if (!parsed) return null;
    labels = parsed;
    i = close + 1;
  }
  while (i < n && (line[i] === " " || line[i] === "\t")) i++;
  const valStart = i;
  while (i < n && line[i] !== " " && line[i] !== "\t") i++;
  const value = parseValueToken(line.slice(valStart, i));
  if (value === null) return null;
  // Anything after the value is an optional millisecond timestamp -- not emitted by this
  // server and not needed to render a scrape, so it is neither parsed nor validated.
  return { name, labels, value };
}

// -- top-level parse ---------------------------------------------------------------------

const MAX_ERROR_LINES = 40;

export function parseMetrics(text: string): ParsedMetrics {
  const help: Record<string, string> = {};
  const type: Record<string, MetricType> = {};
  const samples: Record<string, Sample[]> = {};
  const errors: string[] = [];

  for (const line of text.split(/\r?\n/)) {
    if (line === "") continue;
    if (line.startsWith("# HELP ")) {
      const rest = line.slice(7);
      const sp = rest.indexOf(" ");
      if (sp === -1) help[rest] = "";
      else help[rest.slice(0, sp)] = rest.slice(sp + 1);
      continue;
    }
    if (line.startsWith("# TYPE ")) {
      const rest = line.slice(7);
      const sp = rest.indexOf(" ");
      if (sp !== -1) {
        const kind = rest.slice(sp + 1).trim();
        type[rest.slice(0, sp)] =
          kind === "counter" || kind === "gauge" || kind === "histogram" || kind === "summary"
            ? kind
            : "untyped";
      }
      continue;
    }
    // OpenMetrics' `# EOF` and any other comment this exposition never emits: not this
    // server's fault to explain, so it is skipped rather than logged as an error.
    if (line.startsWith("#")) continue;

    const parsed = parseSampleLine(line);
    if (!parsed) {
      if (errors.length < MAX_ERROR_LINES) {
        errors.push(line.length > 200 ? `${line.slice(0, 200)}…` : line);
      }
      continue;
    }
    (samples[parsed.name] ??= []).push({ labels: parsed.labels, value: parsed.value });
  }

  return { help, type, samples, histograms: assembleHistograms(samples, type), errors };
}

function labelKey(labels: Record<string, string>): string {
  return Object.keys(labels)
    .sort()
    .map((k) => `${k}=${labels[k]}`)
    .join(",");
}

function assembleHistograms(
  samples: Record<string, Sample[]>,
  type: Record<string, MetricType>,
): Record<string, HistogramSeries[]> {
  const result: Record<string, HistogramSeries[]> = {};

  for (const [name, kind] of Object.entries(type)) {
    if (kind !== "histogram") continue;

    const byKey = new Map<string, { labels: Record<string, string>; buckets: HistogramBucket[] }>();
    for (const s of samples[`${name}_bucket`] ?? []) {
      const { le, ...rest } = s.labels;
      if (le === undefined) continue; // a _bucket row with no le is not a bucket row
      const leNum = le === "+Inf" ? Infinity : Number(le);
      if (Number.isNaN(leNum)) continue;
      const key = labelKey(rest);
      let entry = byKey.get(key);
      if (!entry) {
        entry = { labels: rest, buckets: [] };
        byKey.set(key, entry);
      }
      entry.buckets.push({ le: leNum, cumulative: s.value });
    }
    for (const entry of byKey.values()) entry.buckets.sort((a, b) => a.le - b.le);

    const sumByKey = new Map<string, { labels: Record<string, string>; value: number }>();
    for (const s of samples[`${name}_sum`] ?? []) {
      sumByKey.set(labelKey(s.labels), { labels: s.labels, value: s.value });
    }
    const countByKey = new Map<string, { labels: Record<string, string>; value: number }>();
    for (const s of samples[`${name}_count`] ?? []) {
      countByKey.set(labelKey(s.labels), { labels: s.labels, value: s.value });
    }

    const allKeys = new Set<string>([...byKey.keys(), ...sumByKey.keys(), ...countByKey.keys()]);
    if (allKeys.size === 0) continue;

    const list: HistogramSeries[] = [];
    for (const key of allKeys) {
      const b = byKey.get(key);
      const su = sumByKey.get(key);
      const co = countByKey.get(key);
      list.push({
        labels: b?.labels ?? su?.labels ?? co?.labels ?? {},
        buckets: b?.buckets ?? [],
        sum: su ? su.value : null,
        count: co ? co.value : null,
      });
    }
    list.sort((a, b) => labelKey(a.labels).localeCompare(labelKey(b.labels)));
    result[name] = list;
  }

  return result;
}

// -- accessors -------------------------------------------------------------------------

/** All samples of `name` whose labels are a superset of `match` (order-independent). */
export function filterSamples(m: ParsedMetrics, name: string, match: Record<string, string> = {}): Sample[] {
  return (m.samples[name] ?? []).filter((s) => Object.entries(match).every(([k, v]) => s.labels[k] === v));
}

/** The one sample of `name` whose label set is *exactly* `match` -- for unlabelled gauges. */
export function findSample(m: ParsedMetrics, name: string, match: Record<string, string> = {}): number | null {
  const keys = Object.keys(match);
  const found = (m.samples[name] ?? []).find(
    (s) => Object.keys(s.labels).length === keys.length && keys.every((k) => s.labels[k] === match[k]),
  );
  return found ? found.value : null;
}

/** The histogram series of `name` whose labels are a superset of `match`. */
export function findHistogram(m: ParsedMetrics, name: string, match: Record<string, string> = {}): HistogramSeries | null {
  const list = m.histograms[name] ?? [];
  return list.find((h) => Object.entries(match).every(([k, v]) => h.labels[k] === v)) ?? null;
}

// -- quantiles ---------------------------------------------------------------------------

export interface Quantile {
  value: number;
  /**
   * True when the target rank falls in the `+Inf` overflow bucket -- there is no upper edge
   * to interpolate against, so `value` is the last finite boundary and a lower bound, not an
   * estimate. The UI must say "at least", never show it as a number like the others.
   */
  clipped: boolean;
}

/**
 * Linear interpolation within the bucket a quantile's rank falls in -- the same assumption
 * `histogram_quantile` makes, and the same one that makes any Prometheus-bucket quantile an
 * estimate rather than an order statistic: it assumes observations are spread evenly across
 * a bucket's width, which is false whenever the true distribution has structure narrower than
 * a bucket. `LATENCY_BUCKETS_MS` is dense exactly where this project's own measurements say
 * the mass is, which bounds the error without eliminating it.
 */
export function bucketQuantile(h: HistogramSeries, q: number): Quantile | null {
  if (h.buckets.length === 0) return null;
  const total = h.buckets[h.buckets.length - 1].cumulative;
  if (!(total > 0)) return null;

  const rank = q * total;
  let prevLe = 0;
  let prevCum = 0;
  for (const b of h.buckets) {
    if (rank <= b.cumulative) {
      if (b.le === Infinity) return { value: prevLe, clipped: true };
      if (b.cumulative === prevCum) return { value: b.le, clipped: false };
      const frac = (rank - prevCum) / (b.cumulative - prevCum);
      return { value: prevLe + frac * (b.le - prevLe), clipped: false };
    }
    prevLe = b.le;
    prevCum = b.cumulative;
  }
  // Cumulative counts that don't reach `total` by the last bucket mean the buckets are not
  // monotonic -- a malformed or torn scrape. Report the last finite edge as a floor rather
  // than an out-of-range number or a crash.
  const lastFinite = [...h.buckets].reverse().find((b) => b.le !== Infinity);
  return { value: lastFinite ? lastFinite.le : 0, clipped: true };
}
