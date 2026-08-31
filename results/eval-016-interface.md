# Eval 016 — the interface, and whether a stranger would try it

**Date:** 2026-08-30
**Scope:** `ui/` only (source and rebuilt `ui/dist`). No file under `src/`, `tests/`,
`configs/`, `benchmarks/`, `results/*.json`, `README.md`, `ARCHITECTURE.md` or `Makefile` was
touched.
**Reproduce:** `python -m warrant.cli serve --port 8012 --no-warm`, then `ui/dist/index.html`
served at `/`. Every question below was run against that live process, not read out of test
fixtures.

## The problem this addresses

The four screens were correct and unexplorable: an empty search box over 5 CFR, in front of a
visitor who has no idea what the corpus can answer. The engineering — the as-of predicate, the
scope predicate, the evidence-before-prose stream, the failure budget — never got looked at,
because nobody had a reason to type anything.

## What changed

**A live hero on Ask**, above everything else, that states the project's thesis without a
paragraph of prose: one section, one control, and the paragraph that answers the question
changes text under the reader's hand when the control crosses the date that changed it. It
reads `/api/section/575.102` once and resolves the in-force paragraph client-side as the
slider moves, so there is no network wait between the gesture and the text changing — the
thing that makes a control worth actually moving. The amendment date is drawn as a tick on the
control itself, with the label clamped inside the track so it never runs off the edge (this
corpus's newest amendment is also its most recent date, which puts the tick within a few
percent of the right edge — a case the layout has to survive, not a corner case to ignore).

**Five one-click demonstrations** (`ui/src/screens/Ask.tsx`, `DEMOS`), each filling the form
and running the *real* streamed request — nothing staged: two temporal (same section, before
and after its amendment), two scope (same question, `pay_system=GS` vs `FWS`, different
governing part), one exception (evidence that carries its own `except` clause, to watch
whether the generator keeps it).

**The evidence-before-prose asymmetry, drawn rather than stated.** A new `Race` component
(`Ask.tsx`) renders two lanes: evidence fills solid the instant `evidenceAt` is set; prose
hatches with an animated diagonal fill while `phase === "generating"`, because there is no
completion fraction to report honestly against a token budget still being spent, and a
deterministic bar at an invented width would be worse than an honest "still working." Sits
beside the existing millisecond readout rather than replacing it.

**Regulation-set typography.** `splitDesignator` (`ui/src/lib.ts`) peels a leading `(a)`,
`(b)(1)`, `(c)(2)(i)` off a chunk's text; `RegText` (`ui/src/ui.tsx`) hangs it in a fixed
2.5rem gutter instead of running it into the sentence, the way the CFR itself is set. Applied
to evidence rows on Ask. The gutter is reserved even when a chunk has no designator, so a
ledger of mixed rows still lines up.

**The SUPERSEDED stamp, worked harder.** It already sat centred over the text it cancels
(`mix-blend-mode`, `opacity: 0.62`) rather than beside it — that part was already right. Added:
a second, misregistered border (`::before`, offset a few px and 2° off the first) the way a
hand stamp never lands twice in the same place; a dot-grain fill instead of a flat one; a
one-pixel "crack" (`::after`) through the frame where the ink didn't take evenly. `--corner`
variant unaffected.

**Paper grain.** A fixed, self-contained SVG `feTurbulence` texture on `body::before`,
opacity 0.05 light / 0.035 dark, `mix-blend-mode: multiply` / `screen` to match the existing
`--stamp-blend` pattern. No asset, no request — inlined as a data URI.

**One orientation paragraph**, not a wall of text, at the top of Ask: what this answers, that
it cites by evidence id, that a failure names its stage — with a link to Trace for the third
claim.

**Failing and empty states** were already unusually thorough (`ui/src/ui.tsx`'s `explain()`
covers 0/400/404/422/429/503/500 individually, with the *reason* each one exists, not just its
number) and were left as-is rather than reworked; the additions above are net-new surface, not
a rewrite of what was already correct.

Everything above respects the existing rules: `prefers-reduced-motion` collapses all animation
durations to near-zero globally (`styles.css`, pre-existing); the complete light palette lives
on bare `:root`, dark overrides only under the two guarded blocks; no new dependency, no CDN,
no font host.

## Seeded questions, verified live

Server: `python -m warrant.cli serve --port 8012 --no-warm`, RTX 5070 Laptop, warm after the
first generation. Corpus: 13,212 chunk versions, 26 parts, 2017-01-01 → 2026-08-25.

### 1. Temporal — §575.102, "service agreement"

*"How long can a service agreement for a recruitment incentive run?"*

| as of | claim (generated) | cited version |
|---|---|---|
| 2020-06-01 | "The service period may not be less than 6 months and may not exceed 4 years." | `575.110#a@2017-01-01` |
| 2026-08-25 | "The service period may not exceed 4 years." | `575.110#a@2026-02-13` |

§575.102's own definition of "service agreement" changed on 2026-08-25 — the newest amendment
in this corpus — from *"not less than 6 months or more than 4 years"* to *"not more than 4
years."* This replaced the originally-planned example, §630.306 ("by when must restored annual
leave be scheduled"): checked against the live store, that section's own paragraph (a) barely
moves across its 2020-08-10 amendment — the change there is a cross-reference addition
(`, § 630.310(d),`), and the substantive change moved into a *new* section, 630.310, entirely
absent before the amendment. A visitor scrubbing §630.306 alone would see almost no change and
reasonably conclude the demo was broken. §575.102 shows a real, self-contained, one-paragraph
difference and was used for the hero and both `temporal` demo entries instead. (§630.306 is
still a legitimate example of the as-of predicate mattering — the *retrieved set* changes
sharply, per eval-004 and the README — it is just the wrong choice for a single scrubbed
paragraph.)

### 2. Scope — within-grade increases, GS vs FWS

*"How is my within-grade increase determined?"*, `as_of=2026-08-25`

| `pay_system` | top parts retrieved | excluded |
|---|---|---|
| `GS` | 531, 531, 531, 531, 531 | 317, 532, 534 |
| `FWS` | 532, 532, 532, 532, 532 | 317, 511, 531, 534 |

Identical question, disjoint governing parts. Confirms `Scope.of` / `PART_RESTRICTIONS`
actually change which text is admitted, not just which text is labelled.

### 3. Exception — §315.401(b), reinstatement time limit

*"Is there a time limit on reinstatement eligibility after career tenure?"*, `as_of` = corpus
latest

Top evidence: `315.401#b` — *"There is no time limit on the reinstatement eligibility of a
preference eligible or a person who completed the service requirement for career tenure.
Except as provided in paragraph (c) of this section, [...]"*. Live generation split this into
two claims, both grounded and both citing the same version id:

1. *"There is no time limit on the reinstatement eligibility of a preference eligible or a
   person who completed the service requirement for career tenure."*
2. *"Except as provided in paragraph (c) of this section, an agency may reinstate a
   nonpreference eligible who has not completed the service requirement for career tenure only
   within 3 years following the date of separation."*

The generator kept the exception as its own claim rather than dropping it — the interesting
case either way, and now visible rather than asserted.

All five `DEMOS` entries and the hero were re-verified against `/api/ask` (`generate=false`,
retrieval only) after the final build, and the two temporal and one exception entry additionally
against the full `/api/ask/stream` including generation, at the exact dates and scope values
the shipped `ui/src/screens/Ask.tsx` sends.

## Server changes this UI works around

None were made — `ui/` was the only writable surface for this pass — but three are load-bearing
enough to name, because the workaround is duplicated logic that will drift from the server it
is copying:

**1. `/api/ask/stream`'s `evidence` frame has no `section_id`.** `AskResponse.evidence`
(`/api/ask`, non-streamed) carries `section_id` directly; the streamed `StreamEvidence` row
(`serve/api.py`, the `events()` closure inside `ask_stream`) does not, only `chunk_id`
(`630.306#a-2`) and `version_id` (`630.306#a-2@2017-01-01`). The client recovers it by
splitting on `#` and `@` (`sectionOf`/`anchorOf`, `ui/src/lib.ts`), which is correct only
because every `chunk_id` in this corpus happens to be `{section_id}#{anchor}` with no `#` or
`@` inside a section id — true today, unenforced by any type. Adding `section_id` to the
`evidence` SSE payload (one extra field already computed server-side, since `_rows` already
selects the full chunk row) would remove a parsing rule the client has no way to verify against
the server's actual id grammar.

**2. No endpoint enumerates sections.** Timeline's five-item `SUGGESTED` list
(`ui/src/screens/Timeline.tsx`) is hand-picked and static because there is no
`GET /api/sections` or equivalent — the only way the client currently discovers a section id is
by already having one (from evidence, from a claim citation, or from typing it). A lightweight
`/api/sections?part=531` returning `{section_id, heading, part}` (the `_meta()` handler already
runs an equivalent `GROUP BY part` query for `MetaResponse.parts`; a sibling query grouped by
`section_id` is the same shape) would let Timeline offer real search instead of five hardcoded
examples, and would let this report's own demo list be generated rather than hand-verified.

**3. `/api/diff` returns opcodes and a similarity ratio; the wholesale/editorial/substantive
reading is reimplemented client-side.** `Diff.tsx`'s `read()` function duplicates
`corpus/diff.py`'s `WHOLESALE_THRESHOLD = 0.50` and `MIN_CHANGED_TOKENS = 3` as UI constants,
with a comment on both ends noting the duplication. The classification logic that already
exists in `corpus/diff.py` for corpus-build-time use is not called from `serve/api.py`'s
`diff()` handler at all — it recomputes an alignment with a bare `difflib.SequenceMatcher` and
returns only `ops` and `similarity`. Returning `classification: str` on `DiffResponse` (computed
by calling into the same function `corpus/diff.py` already exports) would mean a threshold
changed in one place stops silently disagreeing with the label the UI shows for it — right now
nothing enforces that `WHOLESALE_THRESHOLD` matches between the two files, because they are two
copies of one number in two languages.

None of these block anything the UI does today — all three are read-side workarounds that hold
because of facts about the current corpus (id grammar, a small fixed set of interesting
sections, thresholds that have not moved) rather than because the server guarantees them.

## What was not done

- No new dependency, icon set, or font. Fonts remain the two already self-hosted in
  `ui/src/fonts/`.
- The four screens' navigation, routing (`useRoute`, hash-based) and data model are unchanged.
- `ui/dist` was rebuilt (`npm run build` — `tsc -b && vite build`, 37 modules, no warnings) and
  is committed; `node_modules` remains ignored.
