import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  ApiError,
  api,
  askStream,
  type AskParams,
  type AskResponse,
  type ClaimFrame,
  type ClaimView,
  type ErrorFrame,
  type Evidence,
  type RetrievalFrame,
  type StatusFrame,
  type StreamEvidence,
  type WidenOffer,
} from "./api";

/**
 * One ask, as it arrives.
 *
 * The phases are named after the frames rather than after a spinner, because the ordering is
 * the point: `retrieval` and `evidence` land in tens of milliseconds and `claim` frames land
 * seconds later. A single `loading` boolean over both would erase the one asymmetry this
 * server's design exists to expose.
 */
export type Phase =
  | "idle"
  | "opening"
  | "retrieved"
  | "generating"
  | "done"
  | "refused"
  | "cancelled";

/**
 * One follow-up turn. It reads like a marginal note, so it carries what a note needs: the
 * question, the pinned as-of it was necessarily answered under (never a fresh one), and
 * whichever of the two honest outcomes actually happened.
 */
export interface FollowupTurn {
  id: string;
  question: string;
  status: "asking" | "answered" | "insufficient" | "failed" | "widening";
  traceId: string | null;
  asOf: string;
  scope: string;
  claims: ClaimView[];
  /** Missing references the pinned evidence itself makes. Offered only when insufficient. */
  widen: WidenOffer[];
  /** Chunks fetched by `widen` on this turn, shown here rather than folded into the panel. */
  widened: Evidence[];
  error: (ErrorFrame & { retryAfter?: number | null }) | null;
}

export interface AskState {
  phase: Phase;
  params: AskParams | null;
  retrieval: RetrievalFrame | null;
  evidence: StreamEvidence[];
  status: StatusFrame | null;
  claims: ClaimFrame[];
  done: { trace_id: string | null; generated: boolean; abstained?: boolean; parse_failed?: boolean | null } | null;
  /** An `error` frame, or a status code from before the stream opened. Both are refusals. */
  error: (ErrorFrame & { retryAfter?: number | null }) | null;
  /** Wall clock at which each phase was reached, for the elapsed readouts. */
  openedAt: number | null;
  evidenceAt: number | null;
  finishedAt: number | null;
  /**
   * The stage *output sizes*, which the stream does not carry: the `retrieval` frame has
   * timings and an admitted count, and `AskResponse.trace` has the per-stage counts. Fetched
   * alongside with `generate=false`, which is retrieval-only and read-limited, so it costs a
   * second 18 ms retrieval and nothing else. The Trace screen needs both halves.
   */
  counts: AskResponse["trace"] | null;
  countsError: unknown;
  /** Turns answered from this exchange's pinned evidence, oldest first. */
  followups: FollowupTurn[];
}

const EMPTY: AskState = {
  phase: "idle",
  params: null,
  retrieval: null,
  evidence: [],
  status: null,
  claims: [],
  done: null,
  error: null,
  openedAt: null,
  evidenceAt: null,
  finishedAt: null,
  counts: null,
  countsError: null,
  followups: [],
};

interface AskApi {
  state: AskState;
  run: (params: AskParams) => void;
  cancel: () => void;
  reset: () => void;
  /** Answer `question` from the exchange's current pinned trace. No-op with nothing pinned. */
  askFollowup: (question: string) => void;
  /** Fetch `chunkId` into the pinned set for the turn identified by `turnId`. */
  widen: (turnId: string, chunkId: string) => void;
}

const Ctx = createContext<AskApi | null>(null);

export function AskProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AskState>(EMPTY);
  const abort = useRef<AbortController | null>(null);
  /**
   * The trace id the *next* follow-up pins to. A ref, not state: nothing on screen renders
   * this value directly, and every place that needs it -- `askFollowup`, `widen` -- reads it
   * at call time rather than through a render. It moves forward as each follow-up or widen
   * lands, which is what lets turns chain onto a widened set without re-fetching anything.
   */
  const pinned = useRef<string | null>(null);
  const followupAbort = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    abort.current?.abort();
    abort.current = null;
    setState((s) => (s.phase === "opening" || s.phase === "retrieved" || s.phase === "generating"
      ? { ...s, phase: "cancelled", finishedAt: performance.now() }
      : s));
  }, []);

  const reset = useCallback(() => {
    abort.current?.abort();
    abort.current = null;
    followupAbort.current?.abort();
    followupAbort.current = null;
    pinned.current = null;
    setState(EMPTY);
  }, []);

  const run = useCallback((params: AskParams) => {
    abort.current?.abort();
    followupAbort.current?.abort();
    followupAbort.current = null;
    pinned.current = null;
    const ac = new AbortController();
    abort.current = ac;
    const openedAt = performance.now();
    setState({ ...EMPTY, phase: "opening", params, openedAt });

    // Fired next to the stream rather than after it. Retrieval is the same 18 ms of work
    // either way and this one takes no generation slot, so the Trace screen is complete the
    // moment the evidence is, instead of a screen that has to be "run" a second time.
    api.askTrace(params, ac.signal).then(
      (res) => setState((s) => (s.params === params ? { ...s, counts: res.trace } : s)),
      (err) => {
        if (ac.signal.aborted || (err as Error)?.name === "AbortError") return;
        setState((s) => (s.params === params ? { ...s, countsError: err } : s));
      },
    );

    (async () => {
      try {
        for await (const frame of askStream(params, ac.signal)) {
          if (ac.signal.aborted) return;
          setState((s) => {
            if (s.params !== params) return s;
            switch (frame.type) {
              case "retrieval":
                return { ...s, phase: "retrieved", retrieval: frame.data };
              case "evidence":
                return { ...s, evidence: frame.data, evidenceAt: performance.now() };
              case "status":
                return { ...s, phase: "generating", status: frame.data };
              case "claim":
                return { ...s, claims: [...s.claims, frame.data] };
              case "done":
                // Pin the exchange the moment it exists. A follow-up asked before the user
                // has seen this trace_id is still answering the same evidence -- the id is
                // bookkeeping for chaining requests, not something the user has to copy.
                if (frame.data.trace_id) pinned.current = frame.data.trace_id;
                return { ...s, phase: "done", done: frame.data, finishedAt: performance.now() };
              case "error":
                // An error frame *replaces* done; everything already delivered stays on
                // screen, because evidence that arrived is still evidence.
                return {
                  ...s,
                  phase: "refused",
                  error: frame.data,
                  finishedAt: performance.now(),
                };
            }
          });
        }
        // A stream that ends without `done` or `error` was truncated, and saying so is
        // better than leaving a spinner turning over a connection that is already closed.
        setState((s) =>
          s.params === params && (s.phase === "opening" || s.phase === "retrieved" || s.phase === "generating")
            ? {
                ...s,
                phase: "refused",
                error: { status: 0, detail: "the stream ended without a done frame" },
                finishedAt: performance.now(),
              }
            : s,
        );
      } catch (err) {
        if (ac.signal.aborted || (err as Error)?.name === "AbortError") return;
        const e = err instanceof ApiError ? err : new ApiError(0, String(err));
        setState((s) =>
          s.params === params
            ? {
                ...s,
                phase: "refused",
                error: { status: e.status, detail: e.detail, retryAfter: e.retryAfter },
                finishedAt: performance.now(),
              }
            : s,
        );
      }
    })();
  }, []);

  const askFollowup = useCallback((question: string) => {
    const traceId = pinned.current;
    if (!traceId) return; // nothing pinned yet -- no exchange to follow up on
    const id = crypto.randomUUID();
    const ac = new AbortController();
    followupAbort.current?.abort();
    followupAbort.current = ac;
    setState((s) => ({
      ...s,
      followups: [...s.followups, {
        id, question, status: "asking", traceId: null, asOf: "", scope: "",
        claims: [], widen: [], widened: [], error: null,
      }],
    }));
    api.askFollowup(traceId, question, ac.signal).then(
      (res) => {
        if (ac.signal.aborted) return;
        // Advance the pin only once this turn's own trace exists, so a follow-up asked while
        // this one is still in flight still answers the evidence *this* turn started from.
        if (res.trace_id) pinned.current = res.trace_id;
        setState((s) => ({
          ...s,
          followups: s.followups.map((f) => (f.id === id
            ? { ...f, status: res.kind, traceId: res.trace_id, asOf: res.as_of,
                scope: res.scope, claims: res.claims, widen: res.widen }
            : f)),
        }));
      },
      (err) => {
        if (ac.signal.aborted || (err as Error)?.name === "AbortError") return;
        const e = err instanceof ApiError ? err : new ApiError(0, String(err));
        setState((s) => ({
          ...s,
          followups: s.followups.map((f) => (f.id === id
            ? { ...f, status: "failed",
                error: { status: e.status, detail: e.detail, retryAfter: e.retryAfter } }
            : f)),
        }));
      },
    );
  }, []);

  const widen = useCallback((turnId: string, chunkId: string) => {
    const traceId = pinned.current;
    if (!traceId) return;
    setState((s) => ({
      ...s,
      followups: s.followups.map((f) => (f.id === turnId ? { ...f, status: "widening" } : f)),
    }));
    api.widen(traceId, chunkId).then(
      (res) => {
        if (res.trace_id) pinned.current = res.trace_id;
        setState((s) => ({
          ...s,
          followups: s.followups.map((f) => (f.id === turnId
            ? {
                ...f,
                status: "insufficient",
                widen: f.widen.filter((w) => w.chunk_id !== chunkId),
                widened: [...f.widened, res.added],
              }
            : f)),
        }));
      },
      (err) => {
        const e = err instanceof ApiError ? err : new ApiError(0, String(err));
        setState((s) => ({
          ...s,
          followups: s.followups.map((f) => (f.id === turnId
            ? { ...f, status: "insufficient",
                error: { status: e.status, detail: e.detail, retryAfter: e.retryAfter } }
            : f)),
        }));
      },
    );
  }, []);

  const value = useMemo<AskApi>(
    () => ({ state, run, cancel, reset, askFollowup, widen }),
    [state, run, cancel, reset, askFollowup, widen],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useAsk(): AskApi {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useAsk outside AskProvider");
  return ctx;
}
