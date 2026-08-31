import { useEffect, useMemo, useRef, useState } from "react";
import { useCorpus } from "../app";
import { api, type SectionVersion } from "../api";
import type { Route, Screen } from "../lib";
import {
  addDays,
  badDate,
  day,
  daysBetween,
  inForce,
  isoOf,
  longDate,
  num,
  span,
  today,
  useAsync,
} from "../lib";
import { Bones, Copy, Empty, Failure, Field, SectionLabel, Stamp, Tag } from "../ui";

/**
 * Timeline.
 *
 * `/api/section/{id}` returns every version a section has ever had, with the closed-open
 * interval each one stood as law. The axis draws those intervals to scale rather than as
 * equal steps: three amendments in 2024 and then eight quiet years is a *shape*, and evenly
 * spacing the versions would draw the one picture that hides it.
 */

//: Sections that exist in the shipped eCFR corpus and have something to show. Offered as a
//: starting point only -- the API has no "list the sections" endpoint (see
//: results/eval-016-interface.md), so this cannot be derived, and it is labelled as examples
//: rather than presented as a menu of everything available.
const SUGGESTED: [string, string][] = [
  ["315.201", "Service requirement for career tenure"],
  ["531.603", "Locality pay areas"],
  ["890.302", "Coverage of family members"],
  ["330.609", "Exceptions to CTAP selection priority"],
  ["432.102", "Coverage"],
];

export default function Timeline({
  route,
  go,
}: {
  route: Route;
  go: (screen: Screen, ...rest: string[]) => void;
}) {
  const { meta, clampDate } = useCorpus();
  const routeId = route.rest[0] ?? "";
  const routeDate = route.rest[1] ?? "";

  const [draft, setDraft] = useState(routeId);
  const [asOf, setAsOf] = useState(() =>
    clampDate(badDate(routeDate) ? (meta.latest ?? today()) : routeDate),
  );

  // The hash is the source of truth for *which* section; the date is local, because scrubbing
  // it writes a history entry per day otherwise and the back button stops meaning anything.
  useEffect(() => setDraft(routeId), [routeId]);
  useEffect(() => {
    if (!badDate(routeDate)) setAsOf(clampDate(routeDate));
  }, [routeDate, clampDate]);

  const [section, retry] = useAsync((s) => api.section(routeId, s), [routeId], !!routeId);

  return (
    <div className="screen">
      <div className="screen__head">
        <h1 className="screen__title">Every version, and which one was law.</h1>
        <p className="screen__lede">
          A section is not a document; it is a sequence of documents, each valid over a
          closed-open interval. Scrub the needle and the in-force version changes underneath
          it. Everything the needle has passed carries a stamp.
        </p>
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (draft.trim()) go("timeline", draft.trim());
        }}
      >
        <div className="row" style={{ alignItems: "flex-end" }}>
          <div style={{ flex: "1 1 16rem", minWidth: "12rem" }}>
            <Field
              label="section"
              hint={`5 CFR — ${num(meta.sections)} sections across ${num(meta.part_count)} parts`}
            >
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
          <button className="btn" type="submit" style={{ marginBottom: "1rem" }}>
            open
          </button>
        </div>
      </form>

      <div className="row" style={{ marginTop: "-0.4rem", marginBottom: "1.5rem" }}>
        <span className="label" style={{ paddingBottom: "0.35rem" }}>
          examples
        </span>
        {SUGGESTED.map(([id, heading]) => (
          <button
            key={id}
            className="btn btn--ghost btn--sm"
            style={{ textTransform: "none", letterSpacing: 0 }}
            onClick={() => go("timeline", id)}
            title={heading}
          >
            <span className="mono">{id}</span>
          </button>
        ))}
      </div>

      {!routeId ? (
        <Empty title="No section selected.">
          <p style={{ marginTop: "0.5rem", maxWidth: "42rem" }}>
            Type a section number above, pick an example, or arrive here from a piece of
            evidence on the Ask screen — every evidence row links to the section it came from,
            at the date you asked about.
          </p>
        </Empty>
      ) : section.state === "loading" || section.state === "idle" ? (
        <Bones rows={5} />
      ) : section.state === "failed" ? (
        <Failure error={section.error} onRetry={retry} />
      ) : (
        <History
          sectionId={section.value.section_id}
          heading={section.value.heading}
          part={section.value.part}
          versions={section.value.versions}
          asOf={asOf}
          setAsOf={setAsOf}
          go={go}
        />
      )}
    </div>
  );
}

// -- the axis and the versions --------------------------------------------------------------

function History({
  sectionId,
  heading,
  part,
  versions,
  asOf,
  setAsOf,
  go,
}: {
  sectionId: string;
  heading: string | null;
  part: string;
  versions: SectionVersion[];
  asOf: string;
  setAsOf: (v: string) => void;
  go: (screen: Screen, ...rest: string[]) => void;
}) {
  // The axis spans the section's own life, not the corpus's: a section amended twice in 2024
  // drawn against an eight-year corpus is two hairlines at the right edge.
  const lo = versions[0]?.valid_from ?? today();
  const last = versions[versions.length - 1];
  const hi = last?.valid_to ?? today();
  const total = Math.max(1, daysBetween(lo, hi));
  const offset = Math.min(total, Math.max(0, daysBetween(lo, asOf)));

  const current = versions.findIndex((v) => inForce(v, asOf));
  const bodyRef = useRef<HTMLDivElement>(null);

  const widths = useMemo(
    () =>
      versions.map((v) => {
        const end = v.valid_to ?? hi;
        return Math.max(0.6, (daysBetween(v.valid_from, end) / total) * 100);
      }),
    [versions, hi, total],
  );

  function jump(i: number) {
    setAsOf(versions[i].valid_from);
    // Scroll the version into view rather than only recolouring it: on a section with
    // fourteen versions the recoloured one is usually off-screen.
    window.requestAnimationFrame(() => {
      bodyRef.current
        ?.querySelector(`[data-v="${versions[i].valid_from}"]`)
        ?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    });
  }

  return (
    <>
      <div className="screen__head" style={{ marginBottom: "1.25rem" }}>
        <h2 style={{ fontSize: "var(--t-xl)" }}>
          <span className="mono" style={{ color: "var(--accent)" }}>
            5 CFR § {sectionId}
          </span>
          {heading ? <span style={{ fontStyle: "italic" }}> — {heading}</span> : null}
        </h2>
        <p className="hint" style={{ marginTop: "0.4rem" }}>
          part {part} · {versions.length} version{versions.length === 1 ? "" : "s"} ·{" "}
          {longDate(lo)} → {last?.valid_to ? longDate(last.valid_to) : "in force"}
        </p>
      </div>

      <SectionLabel n="01">amendment history</SectionLabel>
      <div className="scroller">
        <div style={{ position: "relative", minWidth: "34rem", paddingTop: "0.5rem" }}>
          <div className="axis" role="list" aria-label="Versions, drawn to scale">
            {versions.map((v, i) => {
              const stale = v.valid_to !== null && v.valid_to <= today();
              return (
                <button
                  key={v.valid_from}
                  role="listitem"
                  className={
                    "axis__band" +
                    (i === current ? " axis__band--current" : "") +
                    (stale ? " axis__band--superseded" : "")
                  }
                  style={{ width: `${widths[i]}%` }}
                  onClick={() => jump(i)}
                  aria-current={i === current ? "true" : undefined}
                  title={`${v.valid_from} → ${v.valid_to ?? "in force"} · ${span(v.valid_from, v.valid_to)}`}
                >
                  <span className="axis__band-label">{v.valid_from}</span>
                  <span className="axis__band-sub">{span(v.valid_from, v.valid_to)}</span>
                </button>
              );
            })}
            <div
              className="axis__needle"
              style={{ left: `${(offset / total) * 100}%` }}
              aria-hidden="true"
            />
          </div>

          <label className="visually-hidden" htmlFor="scrub">
            As-of date
          </label>
          <input
            id="scrub"
            className="scrub"
            type="range"
            min={0}
            max={total}
            step={1}
            value={offset}
            onChange={(e) => setAsOf(addDays(lo, Number(e.target.value)))}
            aria-valuetext={longDate(asOf)}
          />
          <div
            className="row"
            style={{ justifyContent: "space-between", marginTop: "0.1rem", flexWrap: "nowrap" }}
          >
            <span className="hint" style={{ marginTop: 0 }}>
              {lo}
            </span>
            <span className="hint" style={{ marginTop: 0 }}>
              {isoOf(day(hi))}
            </span>
          </div>
        </div>
      </div>

      <div
        className="row"
        style={{ marginTop: "1rem", alignItems: "center", justifyContent: "space-between" }}
      >
        <div className="row" style={{ alignItems: "baseline" }}>
          <span className="label">as of</span>
          <strong className="mono" style={{ fontSize: "var(--t-lg)" }}>
            {asOf}
          </strong>
          <span className="dim">{longDate(asOf)}</span>
        </div>
        <div className="row">
          <button className="btn btn--ghost btn--sm" onClick={() => setAsOf(lo)}>
            ⇤ first
          </button>
          <button className="btn btn--ghost btn--sm" onClick={() => setAsOf(today())}>
            today ⇥
          </button>
        </div>
      </div>

      {current === -1 ? (
        <div style={{ marginTop: "1rem" }}>
          <Empty title={`Nothing was in force on ${asOf}.`}>
            <p style={{ marginTop: "0.4rem" }}>
              Either the section had not been promulgated yet, or it was removed and not
              replaced. Both are real gaps in the record, not missing data.
            </p>
          </Empty>
        </div>
      ) : null}

      <SectionLabel n="02">the text, version by version</SectionLabel>
      <div ref={bodyRef}>
        {versions.map((v, i) => {
          const isCurrent = i === current;
          const stale = v.valid_to !== null && v.valid_to <= today();
          const prev = versions[i - 1];
          return (
            <article
              key={v.valid_from}
              data-v={v.valid_from}
              className={`version stamped${isCurrent ? " version--current" : ""}`}
            >
              {stale && !isCurrent ? <Stamp /> : null}
              <header className="version__head">
                <div className="row" style={{ alignItems: "baseline", gap: "0.75rem" }}>
                  <span className="version__dates">
                    {v.valid_from} → {v.valid_to ?? "—"}
                  </span>
                  <span className="dim" style={{ fontSize: "var(--t-sm)" }}>
                    {span(v.valid_from, v.valid_to)}
                  </span>
                  {isCurrent ? <Tag kind="accent">in force on {asOf}</Tag> : null}
                  {stale ? <Tag kind="stamp">superseded</Tag> : null}
                  {v.valid_to === null ? <Tag kind="ok">current</Tag> : null}
                </div>
                <div className="row" style={{ gap: "0.4rem" }}>
                  <span className="hint" style={{ marginTop: 0 }}>
                    {v.paragraphs.length} para
                  </span>
                  {prev ? (
                    <button
                      className="btn btn--ghost btn--sm"
                      onClick={() => go("diff", sectionId, prev.valid_from, v.valid_from)}
                    >
                      diff ← {prev.valid_from}
                    </button>
                  ) : null}
                </div>
              </header>
              <div
                className="version__body stamped__text"
                style={stale && !isCurrent ? undefined : { opacity: 1, filter: "none" }}
              >
                {v.heading && v.heading !== heading ? (
                  <p className="hint" style={{ marginTop: 0, marginBottom: "0.6rem" }}>
                    heading at this version: <em>{v.heading}</em>
                  </p>
                ) : null}
                {v.paragraphs.map((p) => (
                  <div className="para" key={p.version_id}>
                    <div className="para__anchor">
                      <Copy value={p.version_id} label={p.anchor ?? "—"} />
                    </div>
                    <div className="para__text">{p.text}</div>
                  </div>
                ))}
              </div>
            </article>
          );
        })}
      </div>
    </>
  );
}
