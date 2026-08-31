/**
 * Typed client for the warrant API.
 *
 * Every type here is transcribed from a Pydantic model in `src/warrant/serve/api.py`, and
 * the transcription is deliberate rather than generated: the server publishes an OpenAPI
 * document at /api/openapi.json, but generating from it at build time would make a clone
 * without a running server unbuildable. When a model changes, this file is the one place
 * that has to change with it.
 *
 * Two things the endpoints do that a naive client gets wrong:
 *
 *   - the question parameter is `q`, not `question` (guard.question_param);
 *   - `/api/ask/stream` returns 200 and *then* reports refusals as an `error` frame, because
 *     by the time admission control refuses, the status code is already spent.
 */

// -- response models -------------------------------------------------------------------

export interface PartSummary {
  part: string;
  sections: number;
  chunks: number;
}

export interface Meta {
  chunks: number;
  sections: number;
  parts: PartSummary[];
  part_count: number;
  /** Null on an empty corpus. The answerable range, not the ingestion floor. */
  earliest: string | null;
  latest: string | null;
  history_floor: string;
  facets: Record<string, string[]>;
  config_hash: string;
  final_k: number;
}

export interface Ready {
  ready: boolean;
  corpus: boolean;
  models: boolean;
  /** `null` means there is no dense index at all -- lexical-only, not "nothing missing". */
  uncovered_chunks: number | null;
  generator: boolean;
  chunks: number | null;
  detail: string | null;
}

export interface Health {
  status: string;
  uptime_s: number;
}

export interface Stage {
  name: string;
  out: number;
}

export interface TraceView {
  admitted: number;
  corpus: number;
  stages: Stage[];
}

export interface Evidence {
  version_id: string;
  chunk_id: string;
  section_id: string;
  anchor: string | null;
  heading: string | null;
  part: string;
  subpart: string | null;
  text: string;
  valid_from: string;
  valid_to: string | null;
}

export interface SpanView {
  start: number;
  end: number;
  score: number;
}

export interface Citation {
  version_id: string;
  span: SpanView | null;
}

export interface ClaimView {
  text: string;
  grounded: boolean;
  citations: Citation[];
}

export interface AskResponse {
  trace_id: string | null;
  question: string;
  as_of: string;
  scope: string;
  excluded_parts: string[];
  trace: TraceView;
  evidence: Evidence[];
  claims: ClaimView[];
  abstained: boolean;
  /** Null when generation did not run -- a different fact from "it ran and parsed". */
  parse_failed: boolean | null;
}

export interface Paragraph {
  anchor: string | null;
  version_id: string;
  text: string;
}

export interface SectionVersion {
  valid_from: string;
  valid_to: string | null;
  heading: string | null;
  part: string;
  paragraphs: Paragraph[];
}

export interface SectionResponse {
  section_id: string;
  heading: string | null;
  part: string;
  versions: SectionVersion[];
}

export interface VersionSummary {
  valid_from: string;
  valid_to: string | null;
  heading: string | null;
  paragraph_count: number;
}

export interface VersionsResponse {
  section_id: string;
  versions: VersionSummary[];
}

export type DiffOpTag = "equal" | "insert" | "delete" | "replace";

export interface DiffOp {
  op: DiffOpTag;
  before: string;
  after: string;
}

export interface DiffResponse {
  section_id: string;
  a: string;
  b: string;
  similarity: number;
  ops: DiffOp[];
}

export interface BudgetStage {
  stage: string;
  failures: number;
  share: string;
}

export interface BudgetRepair {
  repair: string;
  implicated: number;
}

export interface BudgetResponse {
  bucket: string;
  config_hash: string;
  items: number;
  failures: number;
  success_rate: number;
  observational: BudgetStage[];
  interventional: BudgetRepair[];
  stages: string[];
}

// -- SSE frames ------------------------------------------------------------------------

export interface RetrievalFrame {
  admitted: number;
  /** Per-stage wall clock, milliseconds, plus `total`. */
  timings: Record<string, number>;
  excluded_parts: string[];
}

/** The streamed evidence row. Narrower than `Evidence`: no section_id, part or anchor. */
export interface StreamEvidence {
  version_id: string;
  chunk_id: string;
  heading: string | null;
  text: string;
  valid_from: string;
  valid_to: string | null;
  source: string;
  authority: number;
}

export interface StatusFrame {
  stage: string;
  note: string;
}

export interface ClaimFrame {
  text: string;
  grounded: boolean;
  citations: string[];
}

export interface DoneFrame {
  trace_id: string | null;
  generated: boolean;
  abstained?: boolean;
  parse_failed?: boolean | null;
}

export interface ErrorFrame {
  status: number;
  detail: string;
}

export type AskEvent =
  | { type: "retrieval"; data: RetrievalFrame }
  | { type: "evidence"; data: StreamEvidence[] }
  | { type: "status"; data: StatusFrame }
  | { type: "claim"; data: ClaimFrame }
  | { type: "done"; data: DoneFrame }
  | { type: "error"; data: ErrorFrame };

// -- transport -------------------------------------------------------------------------

/**
 * A refusal that carries what the caller needs to act on it.
 *
 * `retryAfter` comes off the header rather than being a constant, because the rate limiter
 * computes it from its own bucket: told to wait 30 s when a token arrives in 2 s, a client
 * wastes 28 s of a ceiling that is already only a few requests a minute.
 */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
    readonly retryAfter: number | null = null,
    readonly problems: { kind: string; detail: string }[] = [],
  ) {
    super(detail);
    this.name = "ApiError";
  }

  /** Whether waiting, unchanged, could plausibly succeed. */
  get transient(): boolean {
    return this.status === 429 || this.status === 503 || this.status === 0;
  }
}

function query(params: Record<string, string | number | boolean | null | undefined>): string {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== null && v !== undefined && v !== "") qs.set(k, String(v));
  }
  const s = qs.toString();
  return s ? `?${s}` : "";
}

function retryAfterOf(res: Response): number | null {
  const raw = res.headers.get("retry-after");
  if (!raw) return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

async function failure(res: Response): Promise<ApiError> {
  let detail = `${res.status} ${res.statusText}`.trim();
  let problems: { kind: string; detail: string }[] = [];
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") {
      detail = body.detail;
    } else if (Array.isArray(body?.detail)) {
      // FastAPI's own 422 shape: a list of {loc, msg}. Naming the parameter matters --
      // "expected an ISO date" is only actionable once you know which of three dates.
      detail = body.detail
        .map((d: { loc?: unknown[]; msg?: string }) =>
          `${(d.loc ?? []).slice(1).join(".") || "request"}: ${d.msg ?? "invalid"}`,
        )
        .join("; ");
    }
    if (Array.isArray(body?.problems)) problems = body.problems;
  } catch {
    /* a body that is not JSON tells us nothing the status has not already said */
  }
  return new ApiError(res.status, detail, retryAfterOf(res), problems);
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { signal, headers: { accept: "application/json" } });
  } catch (err) {
    if ((err as Error)?.name === "AbortError") throw err;
    // Status 0 is "the server is not there", which is a different thing to tell a reader
    // than any status the server chose to send.
    throw new ApiError(0, "the API did not respond; is `warrant serve` running?");
  }
  if (!res.ok) throw await failure(res);
  return (await res.json()) as T;
}

export const api = {
  health: (signal?: AbortSignal) => get<Health>("/health", signal),
  ready: async (signal?: AbortSignal): Promise<Ready> => {
    // /ready answers 503 *with a body* when it is not ready, and that body is the whole
    // point of the endpoint. Treating the status as a failure would discard the diagnosis.
    const res = await fetch("/ready", { signal, headers: { accept: "application/json" } }).catch(
      (err: Error) => {
        if (err?.name === "AbortError") throw err;
        throw new ApiError(0, "the API did not respond; is `warrant serve` running?");
      },
    );
    if (res.status !== 200 && res.status !== 503) throw await failure(res);
    return (await res.json()) as Ready;
  },
  meta: (signal?: AbortSignal) => get<Meta>("/api/meta", signal),
  budget: (signal?: AbortSignal) => get<BudgetResponse>("/api/budget", signal),
  section: (id: string, signal?: AbortSignal) =>
    get<SectionResponse>(`/api/section/${encodeURIComponent(id)}`, signal),
  versions: (id: string, signal?: AbortSignal) =>
    get<VersionsResponse>(`/api/section/${encodeURIComponent(id)}/versions`, signal),
  diff: (sectionId: string, a: string, b: string, signal?: AbortSignal) =>
    get<DiffResponse>(`/api/diff${query({ section_id: sectionId, a, b })}`, signal),

  /**
   * Retrieval only. `generate=false` is read off the raw query string by the rate limiter,
   * so this costs a read token rather than one of the four answers a minute -- it never
   * touches the generation slot, and metering it as though it did once refused 27 of 30
   * requests to a path cheaper than the timeline beside it.
   */
  askTrace: (p: AskParams, signal?: AbortSignal) =>
    get<AskResponse>(
      `/api/ask${query({
        q: p.q,
        as_of: p.as_of,
        pay_system: p.pay_system,
        service: p.service,
        generate: false,
      })}`,
      signal,
    ),
};

export interface AskParams {
  q: string;
  as_of: string;
  pay_system?: string | null;
  service?: string | null;
}

/**
 * Consume `/api/ask/stream`, yielding frames in arrival order.
 *
 * `fetch` plus a hand-rolled parser rather than `EventSource`, for one reason that matters:
 * EventSource reports every pre-stream failure as an untyped `error` with no status. A 422
 * on a malformed date, a 429 from the limiter with its Retry-After, and a dead server would
 * all arrive here as the same blank event, and the UI would have to guess. It also
 * reconnects on its own, which for a ~7 s generation means silently paying for it twice.
 */
export async function* askStream(
  p: AskParams,
  signal?: AbortSignal,
): AsyncGenerator<AskEvent, void, void> {
  const url = `/api/ask/stream${query({
    q: p.q,
    as_of: p.as_of,
    pay_system: p.pay_system,
    service: p.service,
  })}`;

  let res: Response;
  try {
    res = await fetch(url, { signal, headers: { accept: "text/event-stream" } });
  } catch (err) {
    if ((err as Error)?.name === "AbortError") return;
    throw new ApiError(0, "the API did not respond; is `warrant serve` running?");
  }
  // Refusals that happen before the response opens are still status codes, and are the only
  // ones that can be. Everything after this line has to be an `error` frame.
  if (!res.ok) throw await failure(res);
  if (!res.body) throw new ApiError(0, "the stream opened with no body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // Frames are separated by a blank line. Split on it rather than on "\n", because a
      // data payload is one long JSON line and a partial read can land mid-object.
      let cut: number;
      while ((cut = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        const parsed = parseFrame(frame);
        if (parsed) yield parsed;
      }
    }
    const tail = parseFrame(buffer);
    if (tail) yield tail;
  } finally {
    // Cancelling rather than leaving it to GC: an abandoned generation holds the server's
    // one generation slot until it finishes, and this at least closes the socket.
    await reader.cancel().catch(() => undefined);
  }
}

function parseFrame(raw: string): AskEvent | null {
  let event = "";
  const data: string[] = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trim());
  }
  if (!event || data.length === 0) return null;
  let payload: unknown;
  try {
    payload = JSON.parse(data.join("\n"));
  } catch {
    return { type: "error", data: { status: 0, detail: `unparseable ${event} frame` } };
  }
  switch (event) {
    case "retrieval":
      return { type: "retrieval", data: payload as RetrievalFrame };
    case "evidence":
      return { type: "evidence", data: payload as StreamEvidence[] };
    case "status":
      return { type: "status", data: payload as StatusFrame };
    case "claim":
      return { type: "claim", data: payload as ClaimFrame };
    case "done":
      return { type: "done", data: payload as DoneFrame };
    case "error":
      return { type: "error", data: payload as ErrorFrame };
    default:
      // An event name this client has never heard of is a server that moved ahead of it.
      // Dropping it silently is right; crashing on it is not.
      return null;
  }
}
