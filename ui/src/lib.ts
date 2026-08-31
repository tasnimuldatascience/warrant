import { useCallback, useEffect, useRef, useState } from "react";

// -- routing ---------------------------------------------------------------------------

export const SCREENS = ["ask", "timeline", "diff", "trace"] as const;
export type Screen = (typeof SCREENS)[number];

export interface Route {
  screen: Screen;
  /** Everything after the screen: `#/timeline/630.310` -> ["630.310"]. */
  rest: string[];
}

function readHash(): Route {
  const raw = window.location.hash.replace(/^#\/?/, "");
  const parts = raw.split("/").filter(Boolean).map(decodeURIComponent);
  const screen = (SCREENS as readonly string[]).includes(parts[0] ?? "")
    ? (parts[0] as Screen)
    : "ask";
  return { screen, rest: parts.slice(1) };
}

/**
 * Hash routing, and not by preference.
 *
 * `create_app` mounts `/assets` and serves `index.html` at `/` -- and registers no catch-all.
 * A path-routed build would 404 on every deep link the moment it was served by the API it
 * was built for, and would only ever look right under Vite's dev server.
 */
export function useRoute(): [Route, (screen: Screen, ...rest: string[]) => void] {
  const [route, setRoute] = useState<Route>(readHash);
  useEffect(() => {
    const onHash = () => setRoute(readHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const go = useCallback((screen: Screen, ...rest: string[]) => {
    const tail = rest.filter(Boolean).map(encodeURIComponent).join("/");
    window.location.hash = `#/${screen}${tail ? "/" + tail : ""}`;
  }, []);
  return [route, go];
}

// -- dates -----------------------------------------------------------------------------

const DAY = 86_400_000;

/** Parse an ISO date as UTC. Never `new Date("2021-01-01")` plus local-time arithmetic. */
export function day(iso: string): number {
  const [y, m, d] = iso.split("-").map(Number);
  return Date.UTC(y, (m ?? 1) - 1, d ?? 1);
}

export function isoOf(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}

export function daysBetween(a: string, b: string): number {
  return Math.round((day(b) - day(a)) / DAY);
}

export function addDays(iso: string, n: number): string {
  return isoOf(day(iso) + n * DAY);
}

export function today(): string {
  return isoOf(Date.now());
}

const ISO = /^\d{4}-\d{2}-\d{2}$/;

/**
 * The same two checks `api._date` makes, made before the request rather than after it.
 *
 * The shape test alone accepts `2021-13-45`, which the server answers 422 for; doing it here
 * as well is not duplication for its own sake, it is the difference between a field that
 * marks itself invalid as you type and a round trip that spends a rate-limit token to say so.
 */
export function badDate(value: string): string | null {
  if (!ISO.test(value)) return "expected YYYY-MM-DD";
  const [y, m, d] = value.split("-").map(Number);
  const back = new Date(Date.UTC(y, m - 1, d));
  if (back.getUTCFullYear() !== y || back.getUTCMonth() !== m - 1 || back.getUTCDate() !== d) {
    return "not a real calendar date";
  }
  return null;
}

/** "14 March 2023". Long form, because a table of 2023-03-14 next to 2023-04-13 is a wall. */
const LONG = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  month: "long",
  year: "numeric",
  timeZone: "UTC",
});

export function longDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  return LONG.format(new Date(day(iso)));
}

/** "8 years, 7 months" / "3 months" / "14 days". Duration a version stood as law. */
export function span(from: string, to: string | null): string {
  const end = to ?? today();
  const days = Math.max(0, daysBetween(from, end));
  if (days < 62) return `${days} day${days === 1 ? "" : "s"}`;
  const months = Math.round(days / 30.44);
  if (months < 24) return `${months} months`;
  const years = Math.floor(months / 12);
  const rest = months % 12;
  return rest ? `${years} yr ${rest} mo` : `${years} yr`;
}

// -- numbers ---------------------------------------------------------------------------

const N = new Intl.NumberFormat("en-US");
export const num = (n: number): string => N.format(n);

export function ms(v: number): string {
  if (v >= 1000) return `${(v / 1000).toFixed(2)} s`;
  if (v >= 10) return `${v.toFixed(1)} ms`;
  return `${v.toFixed(2)} ms`;
}

export function pct(v: number, digits = 1): string {
  return `${(v * 100).toFixed(digits)}%`;
}

// -- domain ------------------------------------------------------------------------------

/**
 * The section a chunk belongs to, recovered from its id.
 *
 * `/api/ask/stream`'s evidence frame carries `chunk_id` (`630.310#d-1`) and `version_id`
 * (`630.310#d-1@2023-03-14`) but not `section_id` -- unlike the `Evidence` model on
 * `/api/ask`, which does. Splitting here is the price of the streamed shape; see the note in
 * results/eval-016 for the server change that would remove it.
 */
export function sectionOf(chunkId: string): string {
  return chunkId.split("#")[0].split("@")[0];
}

export function anchorOf(chunkId: string): string | null {
  const hash = chunkId.indexOf("#");
  if (hash === -1) return null;
  return chunkId.slice(hash + 1).split("@")[0] || null;
}

/** Whether this version was law on `asOf`. `valid_to` is exclusive. */
export function inForce(v: { valid_from: string; valid_to: string | null }, asOf: string): boolean {
  return v.valid_from <= asOf && (v.valid_to === null || asOf < v.valid_to);
}

/** Whether this version has since been replaced -- the thing the SUPERSEDED stamp marks. */
export function superseded(v: { valid_to: string | null }): boolean {
  return v.valid_to !== null && v.valid_to <= today();
}

/** Tier names from README's authority table. The store records the tier; this labels it. */
export const AUTHORITY: Record<number, string> = {
  1: "5 U.S.C.",
  2: "5 CFR",
  3: "Fed. Register",
  4: "OPM guidance",
  5: "printed record",
};

// -- regulation typography ---------------------------------------------------------------

/**
 * Split a leading paragraph designator off the body of the sentence it introduces.
 *
 * CFR text is not set as run-on prose -- "(a)", "(b)(1)", "(c)(2)(i)" mark a paragraph's
 * place in the outline and are conventionally set in the margin, not run into the line. The
 * pattern is one or more parenthesised alphanumeric groups at the very start of the chunk;
 * eCFR text always begins this way when the chunk boundary is a paragraph.
 */
const DESIGNATOR = /^((?:\([A-Za-z0-9]{1,4}\))+)\s*/;

export function splitDesignator(text: string): { designator: string | null; body: string } {
  const m = DESIGNATOR.exec(text);
  if (!m || !m[1]) return { designator: null, body: text };
  return { designator: m[1], body: text.slice(m[0].length) };
}

// -- react helpers -----------------------------------------------------------------------

export type Async<T> =
  | { state: "idle" }
  | { state: "loading" }
  | { state: "ok"; value: T }
  | { state: "failed"; error: unknown };

/** Run `fn` on mount and whenever `deps` change, aborting the request that is superseded. */
export function useAsync<T>(
  fn: (signal: AbortSignal) => Promise<T>,
  deps: unknown[],
  enabled = true,
): [Async<T>, () => void] {
  const [result, setResult] = useState<Async<T>>({ state: "idle" });
  const [nonce, setNonce] = useState(0);
  const ref = useRef(fn);
  ref.current = fn;

  useEffect(() => {
    if (!enabled) {
      setResult({ state: "idle" });
      return;
    }
    const ac = new AbortController();
    let live = true;
    setResult({ state: "loading" });
    ref.current(ac.signal).then(
      (value) => live && setResult({ state: "ok", value }),
      (error) => {
        if (!live || ac.signal.aborted || (error as Error)?.name === "AbortError") return;
        setResult({ state: "failed", error });
      },
    );
    return () => {
      live = false;
      ac.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, enabled, nonce]);

  return [result, useCallback(() => setNonce((n) => n + 1), [])];
}

/** localStorage that cannot throw. Private windows and blocked site data both raise here. */
export function stored(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function store(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* a remembered preference is a convenience, never a requirement */
  }
}

export function useNow(active: boolean, everyMs = 100): number {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    if (!active) return;
    setNow(performance.now());
    const id = window.setInterval(() => setNow(performance.now()), everyMs);
    return () => window.clearInterval(id);
  }, [active, everyMs]);
  return now;
}
