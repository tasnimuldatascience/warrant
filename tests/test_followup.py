"""Follow-up turns, pinned to one prior exchange's evidence.

The one property worth failing loudly on: a follow-up must never be able to answer from a
different as-of than the turn it follows. `Exchange.as_of` comes from nowhere but the parent
trace, `widen` resolves its target at that same as-of and no other, and `finish_followup`
refuses an answer that cites anything outside the pinned set. Offline throughout -- a
hand-built store and a hand-built `Answer`, no torch, no corpus.
"""

from __future__ import annotations

import pytest

from warrant.generate.answer import Answer, Claim
from warrant.index.store import Chunk, Store
from warrant.observe import TraceStore
from warrant.retrieve.hybrid import Trace
from warrant.serve import followup as fu

T0 = "2020-01-01T00:00:00+00:00"


@pytest.fixture
def store() -> Store:
    with Store(":memory:") as s:
        s.add([
            # Two versions of one paragraph -- the case a follow-up must never blur.
            Chunk(chunk_id="630.306#a", section_id="630.306", title=5, part="630", anchor="a",
                  heading="Restored annual leave",
                  text="annual leave restored must be scheduled within two years, "
                       "except as provided in paragraph (c) of this section",
                  valid_from="2017-01-01", valid_to="2020-08-10"),
            Chunk(chunk_id="630.306#a", section_id="630.306", title=5, part="630", anchor="a",
                  heading="Restored annual leave",
                  text="annual leave restored must be scheduled within three years, "
                       "except as provided in paragraph (c) of this section",
                  valid_from="2020-08-10"),
            # The paragraph (a) refers to, in force across both spans -- the widen target.
            Chunk(chunk_id="630.306#c", section_id="630.306", title=5, part="630", anchor="c",
                  heading="Restored annual leave",
                  text="the head of the agency may extend the deadline in an emergency",
                  valid_from="2017-01-01"),
            # An unrelated section, so an unpinned citation has somewhere real to point at.
            Chunk(chunk_id="531.404#a", section_id="531.404", title=5, part="531", anchor="a",
                  heading="Within-grade increase",
                  text="performance must be at an acceptable level of competence",
                  valid_from="2017-01-01"),
        ], system_from=T0)
        yield s


@pytest.fixture
def traces() -> TraceStore:
    with TraceStore(":memory:") as t:
        yield t


def _record(traces: TraceStore, *, as_of: str, evidence: tuple[str, ...],
           query: str = "by when must restored leave be scheduled?") -> str:
    trace = Trace(query=query, as_of=as_of, scope="government-wide", final=evidence,
                  admitted=len(evidence), config_hash="cfg0001")
    return traces.record(trace)


def _answer(*, evidence: dict[str, str], claims: list[Claim], found: bool = True,
           as_of: str = "2019-01-01") -> Answer:
    return Answer(question="q", as_of=as_of, scope="government-wide", claims=claims,
                 answer_found=found, cited=evidence)


# -- loading the exchange ----------------------------------------------------------------


def test_load_exchange_pins_as_of_scope_and_evidence_from_the_trace(store: Store,
                                                                     traces: TraceStore):
    tid = _record(traces, as_of="2019-01-01", evidence=("630.306#a@2017-01-01",))
    ex = fu.load_exchange(traces, tid)
    assert ex.trace_id == tid
    assert ex.as_of == "2019-01-01"
    assert ex.evidence == ("630.306#a@2017-01-01",)
    assert ex.scope == "government-wide"


def test_an_unknown_trace_id_is_a_named_exception(traces: TraceStore):
    with pytest.raises(fu.NoSuchExchange):
        fu.load_exchange(traces, "never-recorded")


# -- the central guarantee: no drift across the as-of ------------------------------------


def test_excerpts_carry_only_the_version_pinned_by_the_parent_as_of(store: Store,
                                                                     traces: TraceStore):
    """§630.306(a) has two versions. A follow-up pinned to the 2017 trace must see the 2017
    text even though a live query today would resolve to the 2020 one."""
    tid = _record(traces, as_of="2019-01-01", evidence=("630.306#a@2017-01-01",))
    ex = fu.load_exchange(traces, tid)
    excerpts = fu.excerpts_for_exchange(store, ex, "what's the exception?")
    assert len(excerpts) == 1
    vid, _heading, text = excerpts[0]
    assert vid == "630.306#a@2017-01-01"
    assert "two years" in text
    assert "three years" not in text


def test_widen_resolves_at_the_exchange_as_of_not_at_the_other_version(store: Store,
                                                                        traces: TraceStore):
    """The store holds two versions of 630.306#a. Widening an exchange pinned to 2019 must
    never be able to reach the 2020-08-10-forward version, even by naming its chunk_id."""
    tid = _record(traces, as_of="2019-01-01", evidence=())
    ex = fu.load_exchange(traces, tid)
    version_id, evidence = fu.widen(store, ex, "630.306#a")
    assert version_id == "630.306#a@2017-01-01"
    assert evidence == ("630.306#a@2017-01-01",)


def test_widen_after_the_amendment_resolves_the_new_version(store: Store, traces: TraceStore):
    """Same target, later as_of: proves the binding is to `exchange.as_of`, not a constant."""
    tid = _record(traces, as_of="2021-06-01", evidence=())
    ex = fu.load_exchange(traces, tid)
    version_id, _evidence = fu.widen(store, ex, "630.306#a")
    assert version_id == "630.306#a@2020-08-10"


def test_widen_is_a_lookup_not_a_search(store: Store, traces: TraceStore):
    """No address, no chunk -- widen must not fall back to ranking anything."""
    tid = _record(traces, as_of="2019-01-01", evidence=())
    ex = fu.load_exchange(traces, tid)
    with pytest.raises(KeyError):
        fu.widen(store, ex, "999.999#z")


def test_widen_is_a_no_op_when_already_pinned(store: Store, traces: TraceStore):
    tid = _record(traces, as_of="2019-01-01", evidence=("630.306#a@2017-01-01",))
    ex = fu.load_exchange(traces, tid)
    version_id, evidence = fu.widen(store, ex, "630.306#a")
    assert evidence == ex.evidence
    assert version_id == "630.306#a@2017-01-01"


def test_widened_exchange_carries_a_new_trace_id(store: Store, traces: TraceStore):
    tid = _record(traces, as_of="2019-01-01", evidence=())
    ex = fu.load_exchange(traces, tid)
    _vid, widened = fu.widen(store, ex, "630.306#c")
    ex2 = ex.widened("a-new-trace-id", widened)
    assert ex2.trace_id == "a-new-trace-id"
    assert ex2.as_of == ex.as_of
    assert ex2.evidence == widened


# -- the three kinds ------------------------------------------------------------------------


def test_kind_answered_when_the_pinned_evidence_covers_it(store: Store, traces: TraceStore):
    tid = _record(traces, as_of="2019-01-01", evidence=("630.306#a@2017-01-01",))
    ex = fu.load_exchange(traces, tid)
    evidence = {"630.306#a@2017-01-01": "annual leave restored must be scheduled within "
                                        "two years, except as provided in paragraph (c)"}
    answer = _answer(evidence=evidence, claims=[
        Claim(text="Restored leave must be scheduled within two years.",
              evidence=["630.306#a@2017-01-01"])])
    result = fu.finish_followup(ex, "how long do I have?", answer, store=store)
    assert result.kind == "answered"
    assert result.dangling == []


def test_kind_insufficient_names_what_is_missing(store: Store, traces: TraceStore):
    """(a) refers to "paragraph (c)", which was never pinned -- exactly the case `widen`
    exists for."""
    tid = _record(traces, as_of="2019-01-01", evidence=("630.306#a@2017-01-01",))
    ex = fu.load_exchange(traces, tid)
    evidence = {"630.306#a@2017-01-01": "annual leave restored must be scheduled within "
                                        "two years, except as provided in paragraph (c) "
                                        "of this section"}
    answer = _answer(evidence=evidence, claims=[], found=False)
    result = fu.finish_followup(ex, "what's the exception?", answer, store=store)
    assert result.kind == "insufficient"
    assert any(d.status == "missing" and d.target == "630.306#c" for d in result.dangling)
    offers = fu.widenable(result.dangling)
    assert {o["chunk_id"] for o in offers} == {"630.306#c"}


def test_widenable_drops_outside_and_unscoped_and_dedupes(store: Store):
    from warrant.verify.xref import DanglingReference, Reference

    ref = Reference(kind="usc", text="5 U.S.C. 6304", span=(0, 10))
    dangling = [
        DanglingReference(source="a@x", target="5 U.S.C. 6304", status="outside", reference=ref),
        DanglingReference(source="a@x", target="this subpart", status="unscoped"),
        DanglingReference(source="a@x", target="630.306#c", status="missing"),
        DanglingReference(source="b@x", target="630.306#c", status="missing"),
    ]
    offers = fu.widenable(dangling)
    assert offers == [{"chunk_id": "630.306#c", "text": "630.306#c"}]


# -- the guard against citing outside the pinned set --------------------------------------


def test_a_citation_outside_the_pinned_set_is_refused(store: Store, traces: TraceStore):
    """Simulates a generator bug rather than relying on one: `parse_response` already cannot
    produce this, so the only way to exercise the check is to hand it a malformed `Answer`
    directly -- which is the point of having the check at all."""
    tid = _record(traces, as_of="2019-01-01", evidence=("630.306#a@2017-01-01",))
    ex = fu.load_exchange(traces, tid)
    answer = _answer(
        evidence={"630.306#a@2017-01-01": "text"},
        claims=[Claim(text="wrong", evidence=["531.404#a@2017-01-01"])])
    with pytest.raises(fu.PinnedEvidenceViolation):
        fu.finish_followup(ex, "q", answer, store=store)


def test_trace_context_names_the_parent_and_kind(store: Store, traces: TraceStore):
    tid = _record(traces, as_of="2019-01-01", evidence=("630.306#a@2017-01-01",))
    ex = fu.load_exchange(traces, tid)
    ctx = fu.trace_context(ex, "answered")
    assert ctx["parent_trace_id"] == tid
    assert ctx["kind"] == "answered"


def test_followup_trace_is_replayable_like_any_other_request(store: Store, traces: TraceStore):
    """A follow-up records a real `Trace` with `final` set -- traces.record must accept it,
    and reading it back must reproduce the pinned evidence."""
    tid = _record(traces, as_of="2019-01-01", evidence=("630.306#a@2017-01-01",))
    ex = fu.load_exchange(traces, tid)
    evidence = {"630.306#a@2017-01-01": "text"}
    answer = _answer(evidence=evidence,
                     claims=[Claim(text="x", evidence=["630.306#a@2017-01-01"])])
    result = fu.finish_followup(ex, "a follow-up question", answer, store=store)
    child_id = traces.record(result.trace, context=fu.trace_context(ex, result.kind))
    reloaded = traces.load(child_id)
    assert reloaded.as_of == "2019-01-01"
    assert reloaded.final == ["630.306#a@2017-01-01"]
    assert reloaded.context["parent_trace_id"] == tid
