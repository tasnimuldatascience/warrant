import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { ApiError } from "./api";

// -- notes, refusals, empties ------------------------------------------------------------

export function Note({
  kind = "plain",
  label,
  title,
  detail,
  children,
  actions,
}: {
  kind?: "plain" | "bad" | "warn" | "ok" | "accent";
  label: string;
  title: ReactNode;
  detail?: ReactNode;
  children?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className={`note${kind === "plain" ? "" : ` note--${kind}`}`} role="status">
      <span className="note__k">{label}</span>
      <div className="note__body">
        <div className="note__title">{title}</div>
        {detail ? <div className="note__detail">{detail}</div> : null}
        {children}
        {actions ? <div className="note__actions">{actions}</div> : null}
      </div>
    </div>
  );
}

/** Seconds remaining on a Retry-After, counted down rather than restated. */
function useCountdown(from: number | null | undefined): number | null {
  const [left, setLeft] = useState<number | null>(from ?? null);
  useEffect(() => {
    if (from == null) {
      setLeft(null);
      return;
    }
    setLeft(from);
    const id = window.setInterval(
      () => setLeft((v) => (v === null ? null : Math.max(0, v - 1))),
      1000,
    );
    return () => window.clearInterval(id);
  }, [from]);
  return left;
}

/**
 * Every way this API says no, rendered as the same shape.
 *
 * The statuses are not interchangeable and the copy says so: a 429 is this client asking too
 * often, a 503 is the server at a ceiling it measured and published, a 422 is a parameter,
 * and a 500 with `problems` is an answer the server assembled and then refused to stand
 * behind — which is the most interesting one and the easiest to render as a generic failure.
 */
export function Refusal({
  status,
  detail,
  retryAfter,
  problems,
  onRetry,
}: {
  status: number;
  detail: string;
  retryAfter?: number | null;
  problems?: { kind: string; detail: string }[];
  onRetry?: () => void;
}) {
  const left = useCountdown(retryAfter);
  const meta = explain(status);
  return (
    <Note
      kind={meta.kind}
      label={status === 0 ? "no reply" : String(status)}
      title={meta.title}
      detail={detail}
      actions={
        onRetry ? (
          <button
            className="btn btn--ghost btn--sm"
            onClick={onRetry}
            disabled={left !== null && left > 0}
          >
            {left !== null && left > 0 ? `retry in ${left}s` : "retry"}
          </button>
        ) : null
      }
    >
      {meta.why ? <p className="hint">{meta.why}</p> : null}
      {problems?.length ? (
        <ul className="hint" style={{ margin: "0.5rem 0 0", paddingLeft: "1.1rem" }}>
          {problems.map((p, i) => (
            <li key={i}>
              <strong>{p.kind}</strong>: {p.detail}
            </li>
          ))}
        </ul>
      ) : null}
    </Note>
  );
}

function explain(status: number): {
  kind: "bad" | "warn";
  title: string;
  why?: string;
} {
  switch (status) {
    case 0:
      return {
        kind: "bad",
        title: "The API did not answer.",
        why: "Start it with `make serve`, or `python -m warrant.cli serve --no-generate`.",
      };
    case 400:
      return { kind: "warn", title: "The scope was not a scope this corpus knows." };
    case 404:
      return { kind: "warn", title: "Nothing under that address." };
    case 422:
      return { kind: "warn", title: "A parameter did not validate." };
    case 429:
      return {
        kind: "warn",
        title: "Rate limited.",
        why:
          "The answer bucket is four requests a minute with a burst of three; reads are ten " +
          "a second. Retry-After is computed from the bucket, not from a constant.",
      };
    case 503:
      return {
        kind: "warn",
        title: "At capacity, or not ready.",
        why:
          "Generation is serialised at one at a time — a measured 29.2 tok/s and 7.7 " +
          "requests a minute — so over-capacity load is refused rather than accepted and " +
          "abandoned. This is the ceiling being made visible, not a fault.",
      };
    case 500:
      return {
        kind: "bad",
        title: "The server withheld its own answer.",
        why:
          "An assembled answer that fails output validation is not served. The reasons are " +
          "below; the request is recorded either way.",
      };
    default:
      return { kind: "bad", title: "The request failed." };
  }
}

/** Anything thrown by the client, rendered as a refusal. */
export function Failure({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  if (error instanceof ApiError) {
    return (
      <Refusal
        status={error.status}
        detail={error.detail}
        retryAfter={error.retryAfter}
        problems={error.problems}
        onRetry={onRetry}
      />
    );
  }
  return <Refusal status={0} detail={String((error as Error)?.message ?? error)} onRetry={onRetry} />;
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <div className="empty__title">{title}</div>
      {children}
    </div>
  );
}

export function Bones({ rows = 3 }: { rows?: number }) {
  return (
    <div aria-hidden="true" style={{ display: "grid", gap: "0.55rem" }}>
      {Array.from({ length: rows }, (_, i) => (
        <div className="bone" key={i} style={{ width: `${100 - i * 11}%` }} />
      ))}
    </div>
  );
}

// -- marks ---------------------------------------------------------------------------------

export function Stamp({ corner = false, text = "superseded" }: { corner?: boolean; text?: string }) {
  return (
    <span className={`stamp${corner ? " stamp--corner" : ""}`} aria-hidden="true">
      {text}
    </span>
  );
}

export function Tag({
  kind,
  children,
}: {
  kind?: "stamp" | "ok" | "accent" | "warn";
  children: ReactNode;
}) {
  return <span className={`tag${kind ? ` tag--${kind}` : ""}`}>{children}</span>;
}

export function SectionLabel({ n, children }: { n: string; children: ReactNode }) {
  return (
    <h2 className="section-label">
      <span className="section-label__n">{n}</span>
      <span className="label" style={{ color: "var(--ink-2)" }}>
        {children}
      </span>
    </h2>
  );
}

// -- meters ---------------------------------------------------------------------------------

export function Meter({
  label,
  value,
  max,
  display,
  alt = false,
}: {
  label: string;
  value: number;
  max: number;
  display: string;
  alt?: boolean;
}) {
  const w = max > 0 ? Math.max(1.5, (value / max) * 100) : 0;
  return (
    <>
      <span className="meter__k">{label}</span>
      <span className="meter__track">
        <span
          className={`meter__fill${alt ? " meter__fill--alt" : ""}`}
          style={{ width: `${w}%` }}
        />
      </span>
      <span className="meter__v">{display}</span>
    </>
  );
}

// -- copy -----------------------------------------------------------------------------------

export function Copy({ value, label }: { value: string; label?: string }) {
  const [state, setState] = useState<"idle" | "ok" | "no">("idle");
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  return (
    <button
      className="copy"
      title={`copy: ${value}`}
      onClick={() => {
        // navigator.clipboard is unavailable on insecure non-localhost origins, which is
        // exactly how someone reviewing this over a LAN address would see it.
        navigator.clipboard?.writeText(value).then(
          () => setState("ok"),
          () => setState("no"),
        ) ?? setState("no");
        window.clearTimeout(timer.current);
        timer.current = window.setTimeout(() => setState("idle"), 1600);
      }}
    >
      <span aria-hidden="true">{state === "ok" ? "✓" : state === "no" ? "!" : "⧉"}</span>
      <span>{label ?? value}</span>
      <span className="visually-hidden">
        {state === "ok" ? "copied" : state === "no" ? "could not copy" : ""}
      </span>
    </button>
  );
}

// -- form pieces ------------------------------------------------------------------------------

export function Field({
  label,
  hint,
  bad,
  aside,
  children,
}: {
  label: string;
  hint?: ReactNode;
  bad?: string | null;
  aside?: ReactNode;
  children: (id: string) => ReactNode;
}) {
  const id = useId();
  return (
    <div className="field">
      <div className="field__label">
        <label className="label" htmlFor={id}>
          {label}
        </label>
        {aside}
      </div>
      {children(id)}
      {bad ? (
        <div className="hint hint--bad" role="alert">
          {bad}
        </div>
      ) : hint ? (
        <div className="hint">{hint}</div>
      ) : null}
    </div>
  );
}

export function Chips({
  label,
  values,
  value,
  onChange,
  anyLabel = "any",
}: {
  label: string;
  values: string[];
  value: string | null;
  onChange: (v: string | null) => void;
  anyLabel?: string;
}) {
  return (
    <div className="field">
      <div className="field__label">
        <span className="label" id={`${label}-lbl`}>
          {label}
        </span>
      </div>
      <div className="chips" role="group" aria-labelledby={`${label}-lbl`}>
        <button
          type="button"
          className="chip"
          aria-pressed={value === null}
          onClick={() => onChange(null)}
        >
          {anyLabel}
        </button>
        {values.map((v) => (
          <button
            type="button"
            key={v}
            className="chip"
            aria-pressed={value === v}
            onClick={() => onChange(value === v ? null : v)}
          >
            {v}
          </button>
        ))}
      </div>
    </div>
  );
}
