import { useEffect, useMemo, useState } from "react";
import { api, type DiffOp } from "../api";
import type { Route, Screen } from "../lib";
import { longDate, num, pct, span, today, useAsync } from "../lib";
import { Bones, Empty, Failure, Field, Note, SectionLabel, Stamp } from "../ui";

/**
 * Diff.
 *
 * `/api/diff` returns word-level opcodes and a similarity ratio, and nothing else: the
 * *classification* lives in `corpus/diff.py`, which the serving path does not call. So the
 * reading below is done here, against the same two constants that module publishes —
 * WHOLESALE_THRESHOLD 0.50 and MIN_CHANGED_TOKENS 3 — and is labelled as a reading rather
 * than presented as a field the server returned. results/eval-016-interface.md records the
 * server change that would make this unnecessary.
 */

const WHOLESALE_THRESHOLD = 0.5;
const MIN_CHANGED_TOKENS = 3;

/** Editorial once punctuation, case and whitespace are folded out — the same fold diff.py uses. */
function fold(s: string): string {
  return s.toLowerCase().replace(/[^\w\s]/g, "").replace(/\s+/g, " ").trim();
}

interface Reading {
  kind: string;
  tone: "accent" | "warn" | "stamp" | "ok";
  what: string;
}

function read(ops: DiffOp[], similarity: number): Reading {
  const changed = ops
    .filter((o) => o.op !== "equal")
    .reduce((n, o) => n + o.before.split(/\s+/).filter(Boolean).length +
      o.after.split(/\s+/).filter(Boolean).length, 0);
  if (changed === 0) {
    return { kind: "identical", tone: "ok", what: "The two versions carry the same words." };
  }
  const editorialOnly = ops
    .filter((o) => o.op !== "equal")
    .every((o) => fold(o.before) === fold(o.after));
  if (editorialOnly || changed < MIN_CHANGED_TOKENS) {
    return {
      kind: "editorial",
      tone: "warn",
      what:
        "Punctuation, case or whitespace only. A change this small is indistinguishable " +
        "from typographic tidying, so it is not treated as an amendment.",
    };
  }
  if (similarity < WHOLESALE_THRESHOLD) {
    return {
      kind: "wholesale rewrite",
      tone: "stamp",
      what:
        `Similarity ${pct(similarity, 1)} is below the 50% floor: before and after cannot be ` +
        "aligned paragraph by paragraph, so this is a replacement rather than an amendment.",
    };
  }
  return {
    kind: "substantive, localized",
    tone: "accent",
    what:
      `${num(changed)} words moved with the identifier and the surrounding text intact — a ` +
      "real amendment, and the only class this project treats as usable ground truth.",
  };
}

export default function Diff({
  route,
  go,
}: {
  route: Route;
  go: (screen: Screen, ...rest: string[]) => void;
}) {
  const [sectionId, a, b] = [route.rest[0] ?? "", route.rest[1] ?? "", route.rest[2] ?? ""];
  const [draft, setDraft] = useState(sectionId);
  useEffect(() => setDraft(sectionId), [sectionId]);

  const [versions] = useAsync((s) => api.versions(sectionId, s), [sectionId], !!sectionId);
  const list = versions.state === "ok" ? versions.value.versions : [];

  // Two adjacent versions are the interesting default: the amendment, not an arbitrary pair.
  useEffect(() => {
    if (!sectionId || list.length < 2) return;
    if (a && b) return;
    go("diff", sectionId, list[list.length - 2].valid_from, list[list.length - 1].valid_from);
  }, [sectionId, a, b, list, go]);

  const ready = !!(sectionId && a && b && a !== b);
  const [diff, retry] = useAsync((s) => api.diff(sectionId, a, b, s), [sectionId, a, b], ready);

  return (
    <div className="screen">
      <div className="screen__head">
        <h1 className="screen__title">What the words actually became.</h1>
        <p className="screen__lede">
          A word-level alignment of two versions of one section. Deletions are struck in the
          left column and insertions underlined in the right; unchanged runs are folded away
          unless you open them.
        </p>
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (draft.trim()) go("diff", draft.trim());
        }}
      >
        <div className="row" style={{ alignItems: "flex-end" }}>
          <div style={{ flex: "1 1 12rem", minWidth: "10rem" }}>
            <Field label="section">
              {(id) => (
                <input
                  id={id}
                  className="input"
                  value={draft}
                  placeholder="315.201"
                  inputMode="decimal"
                  onChange={(e) => setDraft(e.target.value)}
                />
              )}
            </Field>
          </div>
          {list.length ? (
            <>
              <div style={{ flex: "1 1 11rem", minWidth: "9rem" }}>
                <Field label="before">
                  {(id) => (
                    <select
                      id={id}
                      className="select"
                      value={a}
                      onChange={(e) => go("diff", sectionId, e.target.value, b)}
                    >
                      {list.map((v) => (
                        <option key={v.valid_from} value={v.valid_from}>
                          {v.valid_from} ({v.paragraph_count} para)
                        </option>
                      ))}
                    </select>
                  )}
                </Field>
              </div>
              <div style={{ flex: "1 1 11rem", minWidth: "9rem" }}>
                <Field label="after">
                  {(id) => (
                    <select
                      id={id}
                      className="select"
                      value={b}
                      onChange={(e) => go("diff", sectionId, a, e.target.value)}
                    >
                      {list.map((v) => (
                        <option key={v.valid_from} value={v.valid_from}>
                          {v.valid_from} ({v.paragraph_count} para)
                        </option>
                      ))}
                    </select>
                  )}
                </Field>
              </div>
            </>
          ) : (
            <button className="btn" type="submit" style={{ marginBottom: "1rem" }}>
              load versions
            </button>
          )}
        </div>
      </form>

      {!sectionId ? (
        <Empty title="No section selected.">
          <p style={{ marginTop: "0.5rem", maxWidth: "42rem" }}>
            Enter a section, or arrive here from the Timeline — every version card there
            carries a <span className="mono">diff ←</span> button against the version before
            it.
          </p>
        </Empty>
      ) : versions.state === "failed" ? (
        <Failure error={versions.error} />
      ) : list.length === 1 ? (
        <Note
          kind="plain"
          label="one version"
          title={`§ ${sectionId} has been amended zero times in this corpus.`}
          detail={`In force from ${list[0].valid_from} and unchanged since — there is nothing to compare it against.`}
        />
      ) : !ready ? (
        <Bones rows={4} />
      ) : diff.state === "loading" || diff.state === "idle" ? (
        <Bones rows={6} />
      ) : diff.state === "failed" ? (
        <Failure error={diff.error} onRetry={retry} />
      ) : (
        <Rendered
          sectionId={sectionId}
          a={a}
          b={b}
          ops={diff.value.ops}
          similarity={diff.value.similarity}
          aTo={list.find((v) => v.valid_from === a)?.valid_to ?? null}
          bTo={list.find((v) => v.valid_from === b)?.valid_to ?? null}
        />
      )}
    </div>
  );
}

function Rendered({
  sectionId,
  a,
  b,
  ops,
  similarity,
  aTo,
  bTo,
}: {
  sectionId: string;
  a: string;
  b: string;
  ops: DiffOp[];
  similarity: number;
  aTo: string | null;
  bTo: string | null;
}) {
  const reading = useMemo(() => read(ops, similarity), [ops, similarity]);
  const changes = ops.filter((o) => o.op !== "equal").length;

  return (
    <>
      <SectionLabel n="01">reading</SectionLabel>
      <Note
        kind={reading.tone === "stamp" ? "bad" : reading.tone}
        label={reading.kind}
        title={
          <>
            <span className="mono">§ {sectionId}</span>, {longDate(a)} → {longDate(b)}
          </>
        }
        detail={reading.what}
      >
        <div className="kv" style={{ marginTop: "0.6rem" }}>
          <div className="kv__k">similarity</div>
          <div className="kv__v">{pct(similarity, 2)}</div>
          <div className="kv__k">edit blocks</div>
          <div className="kv__v">
            {num(changes)} of {num(ops.length)} aligned runs
          </div>
        </div>
        <p className="hint" style={{ marginTop: "0.6rem" }}>
          This class is read client-side from the opcodes, against the same thresholds{" "}
          <span className="mono">corpus/diff.py</span> publishes (wholesale below 0.50, fewer
          than 3 changed tokens is editorial). The serving endpoint returns the alignment, not
          the label.
        </p>
      </Note>

      <SectionLabel n="02">side by side</SectionLabel>
      <div className="scroller">
        <div className="diff">
          <div className="diff__cols">
            <div className="diff__colhead stamped" style={{ position: "relative" }}>
              before · {a} → {aTo ?? "—"}{" "}
              <span className="dim">({span(a, aTo)})</span>
              {aTo && aTo <= today() ? <Stamp corner /> : null}
            </div>
            <div className="diff__colhead">
              after · {b} → {bTo ?? "in force"} <span className="dim">({span(b, bTo)})</span>
            </div>
            {ops.map((op, i) => (
              <Row key={i} op={op} />
            ))}
          </div>
        </div>
      </div>
      {changes === 0 ? (
        <p className="hint" style={{ marginTop: "0.8rem" }}>
          Two versions with identical text. The store still holds both, because the fact that
          a re-publication happened is itself part of the record.
        </p>
      ) : null}
    </>
  );
}

/** One aligned run. Equal runs fold; anything else is always open. */
function Row({ op }: { op: DiffOp }) {
  const [open, setOpen] = useState(false);
  const words = op.before.split(/\s+/).filter(Boolean).length;

  if (op.op === "equal") {
    // Short unchanged runs are context and stay; long ones are the bulk of a section and
    // would bury the four words that changed.
    if (words <= 24) {
      return (
        <>
          <div className="diff__cell dim">{op.before}</div>
          <div className="diff__cell dim">{op.after}</div>
        </>
      );
    }
    return (
      <>
        <div className="diff__cell" style={{ gridColumn: "1 / -1", padding: "0 1rem" }}>
          <button className="fold" onClick={() => setOpen(!open)} aria-expanded={open}>
            {open ? "▲ fold" : `▼ ${num(words)} unchanged words`}
          </button>
          {open ? (
            <p className="dim" style={{ padding: "0.5rem 0" }}>
              {op.before}
            </p>
          ) : null}
        </div>
      </>
    );
  }

  return (
    <>
      <div className={`diff__cell${op.before ? "" : " diff__cell--gap"}`}>
        {op.before ? (
          <>
            <span className="diff__op">{op.op === "replace" ? "replaced" : "removed"}</span>
            <del className="d">{op.before}</del>
          </>
        ) : (
          <span className="diff__op">—</span>
        )}
      </div>
      <div className={`diff__cell${op.after ? "" : " diff__cell--gap"}`}>
        {op.after ? (
          <>
            <span className="diff__op">{op.op === "replace" ? "with" : "added"}</span>
            <ins className="d">{op.after}</ins>
          </>
        ) : (
          <span className="diff__op">—</span>
        )}
      </div>
    </>
  );
}

export { read as readChange };
