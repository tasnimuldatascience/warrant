# Eval 022 — follow-ups, pinned to the evidence a turn already retrieved

**Date:** 2026-08-30
**Scope:** `src/warrant/serve/followup.py` (new), `tests/test_followup.py` (new, 14 tests),
`ui/src/ask.tsx`, `ui/src/api.ts`, `ui/src/screens/Ask.tsx`, `ui/src/styles.css`, rebuilt
`ui/dist`. No file under `serve/api.py`, `serve/cache.py`, `serve/guard.py`, `generate/*`,
`retrieve/*`, `verify/*`, `eval/*`, `corpus/*`, `index/store.py`, `sources/*`, `cli.py`,
`config.py` or `train/*` was touched — `followup.py` exposes what `api.py` needs and the exact
wiring is below, not applied here.
**Reproduce:** `python -m warrant.cli serve --port 8014 --no-warm` for the interface;
`python -m pytest tests/test_followup.py -q` for the offline suite; the live checks below were
run against the real corpus and trace store on this machine, not synthetic fixtures.

## The problem

Warrant is single-shot by design: question, as-of, scope in; evidence and cited claims out.
That is not an interface that is missing a feature — the README measures the as-of predicate
at **+96.1 points of wrong-version rate, 220 wins / 0 losses, p=1.2e-66**, specifically because
it is applied fresh to every request rather than carried forward. A chat window that lets a
user ask "and what's the exception?" after "by when must restored leave be scheduled?" has
exactly two ways to answer the second question, and both are wrong by default: carry the first
question's date forward silently (the user never learns which law the second answer came from),
or re-resolve the date to today (two turns now cite two different regulations without saying
so). Building a follow-up feature that does either would reintroduce, as the default behaviour,
the precise failure the as-of predicate exists to prevent.

## The design

**An exchange is a trace, not a new kind of state.** `serve/api.py` already records every
request as a `retrieve.hybrid.Trace` (`observe/trace_store.py`) and hands the caller its
`trace_id`. `followup.Exchange` is reconstructed from that trace — question, resolved `as_of`,
scope, and the pinned evidence version ids — rather than kept in a second, in-memory session
store. Nothing new has to be invented to remember what a turn saw, nothing expires when the
process restarts, and a follow-up is exactly as replayable as the request it follows because it
*is* one: `followup.exchange_trace` builds a real `Trace` (only the `final` stage populated)
that `TraceStore.record` persists like any other.

**No parameter for the one thing that would break the guarantee.** A follow-up endpoint takes a
`trace_id` and a question — nothing else. There is no `as_of` or scope parameter anywhere in
`followup.py`'s public surface, so there is no place through which a later turn could smuggle in
a different date. `Exchange.as_of` comes from the parent trace and from nowhere else. Changing
the date or scope is not something a follow-up can express; it is a new question, answered by a
new `/api/ask`.

**Three kinds of turn, and this module implements two of them:**

1. *Answerable from the pinned evidence* — `finish_followup` returns `kind="answered"`.
2. *Needs evidence the set does not hold* — the generator abstains, `kind="insufficient"`, and
   `verify.xref.dangling_references` is run over the pinned excerpts to find what they refer to
   that was never retrieved. `widenable()` filters that to targets the corpus actually holds
   (`status == "missing"`) and dedupes by chunk id — `outside` (a title-5-U.S.C. or other
   non-chapter-I reference) and `unscoped` ("this subpart") are reported nowhere near a button,
   because neither names anything `widen` could fetch.
3. *Changes the date or scope* — enforced by omission, described above, not by a runtime check.

**`widen` is a lookup, never a search.** Given a chunk id (the exact address a dangling
reference already resolved to), it is a single indexed query — `chunk_id = ? AND valid_from <=
exchange.as_of AND ...` — bound to the exchange's own `as_of`, never to today. That binding is
the whole reason `widen` is safe to offer at all: a version-agnostic "fetch this chunk" button
would be exactly the kind of silent re-interpretation the as-of predicate exists to prevent, and
binding the lookup to `exchange.as_of` instead of to "now" is what keeps it from becoming one.

**Every claim is checked against the pinned set, not just offered it.** `finish_followup` calls
`_check_pinned`, which raises `PinnedEvidenceViolation` if any claim cites a version id outside
`exchange.evidence`. This is defense in depth, not the only thing standing in the way —
`generate.answer.parse_response` can only resolve an excerpt *number* back to a version id that
was actually offered, so there is structurally no id for a well-behaved generator to
hallucinate — but the check exists so the invariant is asserted, not merely argued, the same way
`serve.guard.validate_answer` checks the live path's version of it after generation.

## What this reuses, and what it does not touch

`Retriever` is not imported by `followup.py` at all — a follow-up never runs the lexical, dense,
fusion or rerank stages. `Generator` is not imported either: `finish_followup` takes an
already-produced `Answer`, because the model call belongs to whoever already owns the generation
slot and its deadline (`api._generate_answer`), and re-implementing that here would be exactly
the second pipeline the brief rules out. `excerpts_for_exchange` delegates to
`generate.answer.excerpts_for` unmodified, by constructing a real `Trace` with only `final` set
— not a parallel implementation that reads the same rows a different way.

## A real bug this caught before it shipped

The first version of `excerpts_for_exchange` defaulted its `limit` to
`generate.answer.MAX_CONTEXT_CHUNKS` (16), matching the ordinary retrieval path. Live-tested
against the real corpus at a 16-chunk exchange, `widen` correctly appended a 17th chunk to
`exchange.evidence` — and `excerpts_for_exchange` then silently truncated it straight back off,
because it re-sliced the same 16-chunk window before the caller ever saw the addition. That is
precisely the silent failure this module exists to rule out, reintroduced by an off-the-shelf
default. The fix: `excerpts_for_exchange` now defaults to the size of `exchange.evidence`
itself, and the generation-context cap is left to `serve.guard.bound_excerpts`, which the caller
already runs downstream and which reports what it drops (`Prompt.dropped`) instead of dropping
it before anyone could know. Caught by the live check in the next section, not by the offline
suite, which used pinned sets small enough never to hit the cap — worth recording as a gap in
`tests/test_followup.py` if this module is extended.

## Verified live, against the real corpus and trace store

`data/warrant.sqlite3` (schema v4, 13,212 chunk versions), read through
`warrant.serve.api.Runtime` exactly as `/api/ask` builds it, with real `Retriever.retrieve` and
a real `TraceStore`. Real generation was not reachable in this session — the GPU already held
~7.8/8.1 GB for the developer's own server on `:8000`, and the CPU fallback for the 1.5B model
exceeded this machine's paging file — so the generator's output was stood in with hand-built
`Answer` objects, exactly the seam `test_followup.py` exercises offline and the same substitution
`test_api.py` makes for its own generation-adjacent paths. Everything upstream and downstream of
that seam — retrieval, trace recording and reloading, evidence pinning, cross-reference
detection, `widen` resolution — ran against real data:

- **§630.306 straddles its real 2020-08-10 amendment.** Retrieving "by when must restored leave
  be scheduled?" as of `2019-01-01` and `2022-01-01` returns `630.306#a@2017-01-01` and
  `630.306#a@2020-08-10` respectively — the two trace ids were loaded as independent exchanges,
  and neither exchange's pinned evidence ever contained the other's version of `630.306#a` (five
  chunks from unrelated, unamended sections were legitimately shared between them).
- **A follow-up's excerpts equal its exchange's pinned set exactly**, both directions: no chunk
  outside `exchange.evidence` appeared, and every id in `exchange.evidence` was fetched.
  `excerpts_for_exchange`'s output was byte-identical to calling
  `generate.answer.excerpts_for` directly on the same trace.
- **Cross-reference detection on real text**, on a 16-chunk pinned set for the 2019 exchange:
  `dangling_references` found **9** references the pinned chunks make that the pinned set does
  not satisfy — `630.306#a` referring to `630.306#c` ("paragraphs (b) and (c) of this section"),
  `630.308#a` to `630.308#b` and to `630.310`, `630.309#a` to `630.309#b`, `630.311#c` and
  `630.311#d` both to `630.311#c-2`, `630.311#d` also to `630.311#a` and `630.311#b`, and
  `630.911#a` to `630.911#b`. `widenable()` collapsed these to **8** distinct offers (deduping
  the doubly-cited `630.311#c-2`), all `status == "missing"` — none were `outside` or
  `unscoped` on this set, so all eight were real, actionable fetch targets.
- **`widen` resolved at the exchange's own as-of, verified against a direct query, not against
  its own arithmetic**: fetching `630.306#c` for the 2019 exchange returned
  `630.306#c@2017-01-01`, matched independently by a plain `SELECT ... WHERE valid_from <= ?
  AND (valid_to IS NULL OR valid_to > ?)` against `2019-01-01`. The same target resolved to
  `630.306#a@2020-08-10` when looked up from an exchange pinned at `2021-06-01` — proving the
  binding is to `exchange.as_of`, not to a constant.
- **A malformed `Answer` citing an unpinned version id was refused** with
  `PinnedEvidenceViolation`, on real chunk ids from the real store.
- **An unknown trace id raised `NoSuchExchange`**, not a bare `KeyError`.
- **Follow-up traces round-tripped**: recording a follow-up's `Trace` and reloading it by its
  own new trace id reproduced the exact pinned `as_of` and evidence set, with
  `parent_trace_id` and `kind` intact in the `context` column.

## The interface

`Ask.tsx` gained a fourth section, `follow-up`, shown once an exchange has a trace and
non-empty evidence — not before, and never as a chat window. No bubbles, no avatars, no typing
indicator: a turn is an `<article className="followup">` with a rule down the left margin (`2px
solid var(--rule-firm)`), reading as a note in the margin of the record rather than a message
from a second speaker. Every turn restates its pinned as-of in mono
(`ui/src/screens/Ask.tsx`'s `FollowupItem`), and the section header above the form states the
constraint in prose: *"a different date or scope is a new question, not a follow-up — change the
form above and ask again."* Widened chunks render inline on the turn that fetched them
(`WidenedRow`, reusing the `.ev` row styling from the evidence ledger with a `fetched by widen`
tag) rather than being merged into the evidence panel above it, so a turn that reached outside
the shared context says so instead of quietly redrawing the ledger every other turn already
read.

State lives in `ask.tsx`'s `AskProvider`: `AskState.followups` is the list rendered, and a
`pinned` ref (not React state — nothing renders it directly) tracks which trace id the *next*
follow-up or widen call pins to, advancing forward as each one lands. Starting a new `/api/ask`
(`run()`) or clearing the screen (`reset()`) resets it to `null` along with everything else, so
old follow-ups do not survive a genuinely new question — the visible mechanism for kind 3.

## Exact `api.py` wiring

Two endpoints and two small, backward-compatible extensions. `followup.py` exposes everything
functional; nothing below invents new logic, it only calls what's already public — `Runtime`,
`_generate_answer`, `_record`, `excerpts_for`, `guard.bound_excerpts`,
`guard.check_answer_against` — the same way `ask()` already does.

**1. `_record` needs one new optional parameter**, so a follow-up or widen trace can carry
`parent_trace_id` and `kind` in its `context` column:

```python
def _record(rt: Runtime, trace: Any, answer: Any = None, context: Any = None) -> str | None:
    ...
    tid = traces.record(trace, answer=payload, context=context)   # was: traces.record(trace, answer=payload)
```

No existing call site changes (`context` defaults to `None`).

**2. Two new response models**, alongside the existing ones:

```python
class WidenOffer(BaseModel):
    chunk_id: str
    text: str


class FollowupResponse(BaseModel):
    trace_id: str | None = None
    parent_trace_id: str
    question: str
    as_of: str
    scope: str
    kind: str                     # "answered" | "insufficient"
    abstained: bool
    parse_failed: bool | None
    claims: list[ClaimView] = Field(default_factory=list)
    widen: list[WidenOffer] = Field(default_factory=list)


class WidenResponse(BaseModel):
    trace_id: str | None = None
    parent_trace_id: str
    added: Evidence
    pinned_count: int
```

**3. Two endpoints**, plus an import and one middleware tweak:

```python
from . import followup
```

```python
@app.get("/api/ask/followup", response_model=FollowupResponse)
async def ask_followup(request: Request, trace_id: str = Query(max_length=64),
                       question: guard.Question = guard.QuestionParam) -> FollowupResponse:
    if not rt.generate:
        raise HTTPException(503, "generation is off on this server")
    q = question.text

    def _load():
        rt.read_only()
        exchange = followup.load_exchange(rt.traces, trace_id)
        return exchange, followup.excerpts_for_exchange(rt.store, exchange, q)

    try:
        exchange, excerpts = await anyio.to_thread.run_sync(_load)
    except followup.NoSuchExchange as exc:
        raise HTTPException(404, f"no such exchange: {exc}") from exc
    if not excerpts:
        raise HTTPException(409, "the parent turn retrieved no evidence to follow up from")

    prompt = guard.bound_excerpts(excerpts)
    prompt.cost().check(GENERATE_DEADLINE_S)
    answer = await _generate_answer(rt, q, prompt.excerpts, as_of=exchange.as_of,
                                    scope=exchange.scope, deadline=request.state.deadline)
    result = followup.finish_followup(exchange, q, answer, store=rt.store)
    guard.check_answer_against(rt.store, answer, as_of=exchange.as_of,
                               retrieved=[v for v, _, _ in prompt.excerpts])
    tid = _record(rt, result.trace, answer=answer,
                 context=followup.trace_context(exchange, result.kind))
    return FollowupResponse(
        trace_id=tid, parent_trace_id=exchange.trace_id, question=q, as_of=exchange.as_of,
        scope=exchange.scope, kind=result.kind, abstained=answer.abstained,
        parse_failed=answer.parse_failed,
        claims=[ClaimView(text=c.text, grounded=c.grounded, citations=[
            Citation(version_id=vid, span=None if sp is None else
                    SpanView(start=sp.start, end=sp.end, score=round(sp.score, 3)))
            for vid, sp in c.spans.items()]) for c in answer.claims],
        widen=[WidenOffer(**o) for o in followup.widenable(result.dangling)],
    )


@app.get("/api/ask/widen", response_model=WidenResponse)
def ask_widen(trace_id: str = Query(max_length=64),
             chunk_id: str = Query(max_length=64)) -> WidenResponse:
    rt.read_only()
    try:
        exchange = followup.load_exchange(rt.traces, trace_id)
    except followup.NoSuchExchange as exc:
        raise HTTPException(404, f"no such exchange: {exc}") from exc
    try:
        version_id, widened = followup.widen(rt.store, exchange, chunk_id)
    except KeyError as exc:
        raise HTTPException(
            404, f"nothing by {chunk_id!r} was in force on {exchange.as_of}") from exc
    trace = followup.exchange_trace(exchange.widened("", widened), exchange.question)
    tid = _record(rt, trace, context=followup.trace_context(exchange, "widen", chunk_id=chunk_id))
    row = _rows(rt.store, [version_id])[0]
    return WidenResponse(trace_id=tid, parent_trace_id=exchange.trace_id,
                         added=_evidence(row), pinned_count=len(widened))
```

**4. Rate limiting**: `ask_followup` costs a generation slot exactly like `/api/ask`, so it
belongs in the answer bucket rather than the read one. In `create_app`'s
`app.add_middleware(guard.RateLimitMiddleware, ...)` call, add
`answer_paths=("/api/ask", "/api/ask/followup")`. `ask_widen` is a single indexed lookup — no
generation, no ranking — and is correctly left in the default read bucket by not appearing in
that tuple.

## What this deliberately does not build

**No long-term memory.** An `Exchange` lives exactly as long as its trace. There is no
per-user, per-session or cross-exchange store — `followup.py` imports no session concept and
writes nothing keyed on anything but a trace id the caller already had.

**No cross-session state.** Two browser tabs, or the same tab reloaded, share nothing: the
`trace_id` a follow-up pins to lives in the client's own `AskState`, not in a cookie or a
server-side session table. Losing it (closing the tab) does not corrupt anything server-side —
it just means the next follow-up has no `trace_id` to send, and there is no follow-up form to
show without one.

**No personalisation.** `widen` offers the same targets to any caller looking at the same
evidence; nothing in this module reads who is asking. The project's own position on this
(`README.md`, "What this is not") is that filtering by who is asking is an *applicability*
question, answered by the scope predicate that already exists — never a per-user preference
layered on top of retrieval.

These are not gaps waiting to be filled in a later pass — a follow-up that remembered a user
across exchanges, or personalised what it offered to widen, would be exactly the kind of state
this whole feature exists to keep out of a system whose entire premise is that the same question
must get the same answer to anyone asking it as of the same date.
