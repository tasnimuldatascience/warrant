"""Reference-directed expansion: what it admits, what it refuses, and where it stops.

No models and no network. Every test builds the ranked pool by hand rather than letting BM25
produce it, because what is under test is the walk -- which reference is followed, which is
refused by a predicate, and what happens when the citations form a loop -- and a test that
also depends on a lexical ranking fails for two reasons at once.

The three that matter are at the bottom: the cycle, the reference into a superseded version,
and the reference into a part the asker's scope excludes.
"""

from __future__ import annotations

import pytest

from warrant.index.store import Chunk, Store
from warrant.retrieve.hybrid import Candidate, Retriever
from warrant.retrieve.multihop import (
    HOP_STAGE,
    TRACE_ATTR,
    MultiHopRetriever,
    ReferenceExpander,
)
from warrant.retrieve.scope import GOVERNMENT_WIDE, Scope

T0 = "2026-01-01T00:00:00+00:00"
PARTS = ["300", "531", "630", "890"]


@pytest.fixture
def store() -> Store:
    """A corpus whose text points at itself the way the real one does.

    §630.306(a) excepts (b); (b) points on to §630.310; §630.310(b) points back at
    §630.306(a). That triangle is not contrived -- it is the shape eval-006 §1 found on the
    restored-leave sections, and it is what a walk with no visited set runs forever on.
    """
    with Store(":memory:") as s:
        s.add([
            Chunk(chunk_id="630.306#a", section_id="630.306", title=5, part="630", anchor="a",
                  heading="Time limit for use of restored annual leave",
                  text="Except as provided in paragraph (b) of this section, annual leave "
                       "restored under this section must be scheduled and used within "
                       "2 years.",
                  valid_from="2017-01-01", valid_to="2020-08-10"),
            Chunk(chunk_id="630.306#a", section_id="630.306", title=5, part="630", anchor="a",
                  heading="Time limit for use of restored annual leave",
                  text="Except as provided in paragraph (b) of this section, annual leave "
                       "restored under this section must be scheduled and used within "
                       "3 years.",
                  valid_from="2020-08-10"),
            Chunk(chunk_id="630.306#b", section_id="630.306", title=5, part="630", anchor="b",
                  heading="Time limit for use of restored annual leave",
                  text="An employee described in paragraph (a) of this section may schedule "
                       "restored leave as provided in § 630.310.",
                  valid_from="2017-01-01"),
            Chunk(chunk_id="630.310#a", section_id="630.310", title=5, part="630", anchor="a",
                  heading="Restoration of annual leave",
                  text="General provisions govern the restoration of forfeited annual leave.",
                  valid_from="2017-01-01"),
            Chunk(chunk_id="630.310#b", section_id="630.310", title=5, part="630", anchor="b",
                  heading="Restoration of annual leave",
                  text="Restored leave is forfeited unless it is used as provided in "
                       "§ 630.306(a).",
                  valid_from="2017-01-01"),
            Chunk(chunk_id="630.201#a", section_id="630.201", title=5, part="630", anchor="a",
                  heading="Definitions",
                  text="Accrual of annual leave is computed under 5 CFR 531.404(a).",
                  valid_from="2017-01-01"),
            Chunk(chunk_id="531.404#a", section_id="531.404", title=5, part="531", anchor="a",
                  heading="Earning within-grade increases",
                  text="An employee earns a within-grade increase at an acceptable level of "
                       "competence.",
                  valid_from="2017-01-01"),
            # §890.102 is written with paragraph (j)'s chapeau running into (j)(1), so the
            # store holds j-1 and no bare #j -- the addressing artefact resolve() walks for.
            Chunk(chunk_id="890.102#j-1", section_id="890.102", title=5, part="890",
                  anchor="j-1", heading="Coverage",
                  text="Coverage begins on the first day of the first pay period.",
                  valid_from="2017-01-01"),
            Chunk(chunk_id="890.103#a", section_id="890.103", title=5, part="890", anchor="a",
                  heading="Correction of enrollment",
                  text="Enrollment continues as provided in paragraph (j) of § 890.102.",
                  valid_from="2017-01-01"),
            Chunk(chunk_id="300.101#a", section_id="300.101", title=5, part="300", anchor="a",
                  heading="Purpose",
                  text="The head of an agency may grant an employee excused absence.",
                  valid_from="2017-01-01"),
        ], system_from=T0)
        yield s


def expander(store: Store, **kw) -> ReferenceExpander:
    return ReferenceExpander(store, **kw)


def pool(*version_ids: str) -> list[Candidate]:
    return [Candidate(v, rank=i) for i, v in enumerate(version_ids, start=1)]


def ids(expansion) -> list[str]:
    return [c.version_id for c in expansion.final]


# -- admission -------------------------------------------------------------------


def test_follows_the_exception_the_cited_paragraph_names(store: Store):
    """The motivating failure: (a) says "except as provided in (b)" and (b) is not shown."""
    e = expander(store, budget=1).expand(pool("630.306#a@2017-01-01"), final_k=2,
                                         as_of="2018-01-01")
    assert ids(e) == ["630.306#a@2017-01-01", "630.306#b@2017-01-01"]
    assert e.admitted[0].reference == "paragraph (b) of this section"
    assert e.admitted[0].source == "630.306#a@2017-01-01"
    assert e.admitted[0].depth == 2


def test_a_closed_evidence_set_is_left_exactly_as_it_was(store: Store):
    """A query whose references are already satisfied must retrieve what it retrieves today.

    Unfilled budget is handed back rather than shortening the answer, so this is the same
    list the first hop would have produced, in the same order.
    """
    base = pool("300.101#a@2017-01-01", "630.310#a@2017-01-01")
    e = expander(store, budget=4).expand(base, final_k=2, as_of="2018-01-01")
    assert ids(e) == ["300.101#a@2017-01-01", "630.310#a@2017-01-01"]
    assert e.admitted == ()


def test_a_target_already_in_the_set_is_not_admitted_twice(store: Store):
    e = expander(store, budget=2).expand(
        pool("630.306#a@2017-01-01", "630.306#b@2017-01-01"), final_k=4, as_of="2018-01-01")
    assert ids(e).count("630.306#b@2017-01-01") == 1
    # (b) is present, so (a)'s exception is not dangling; (b)'s own § 630.310 still is.
    assert [a.target for a in e.admitted] == ["630.310"]


def test_a_section_level_reference_admits_the_head_of_that_section(store: Store):
    """"as provided in § 630.310" names no paragraph, so the top of the section is admitted."""
    e = expander(store, budget=2).expand(pool("630.306#b@2017-01-01"), final_k=3,
                                         as_of="2018-01-01")
    section = next(a for a in e.admitted if a.kind == "section")
    assert (section.target, section.chunk_id) == ("630.310", "630.310#a")


def test_a_reference_to_an_inlined_chapeau_resolves_to_the_paragraph_that_exists(store: Store):
    """§890.102 holds j-1 and no bare #j. "paragraph (j)" must not be called unresolvable."""
    e = expander(store, budget=1).expand(pool("890.103#a@2017-01-01"), final_k=2,
                                         as_of="2018-01-01")
    assert e.admitted[0].target == "890.102#j"
    assert e.admitted[0].chunk_id == "890.102#j-1"


def test_the_walk_is_deterministic(store: Store):
    base = pool("630.306#a@2017-01-01", "630.310#b@2017-01-01")
    runs = [ids(expander(store, budget=2).expand(base, final_k=4, as_of="2018-01-01"))
            for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_references_of_the_better_ranked_chunk_are_followed_first(store: Store):
    """The ordering policy, and the only one: no score is invented for a hop-2 candidate."""
    high = expander(store, budget=1).expand(
        pool("630.306#a@2017-01-01", "890.103#a@2017-01-01"), final_k=3, as_of="2018-01-01")
    low = expander(store, budget=1).expand(
        pool("890.103#a@2017-01-01", "630.306#a@2017-01-01"), final_k=3, as_of="2018-01-01")
    assert high.admitted[0].chunk_id == "630.306#b"
    assert low.admitted[0].chunk_id == "890.102#j-1"


# -- the budget ------------------------------------------------------------------


def test_the_budget_caps_admissions_and_displaces_the_tail(store: Store):
    base = pool("630.306#a@2017-01-01", "890.103#a@2017-01-01", "300.101#a@2017-01-01")
    e = expander(store, budget=1).expand(base, final_k=3, as_of="2018-01-01")
    assert len(e.admitted) == 1
    assert len(e.final) == 3
    # The third slot was the tail of the first hop and is now the reference.
    assert ids(e) == ["630.306#a@2017-01-01", "890.103#a@2017-01-01", "630.306#b@2017-01-01"]
    assert e.dangling >= 2      # 890.103 pointed somewhere too; the budget refused it


def test_a_zero_budget_is_the_shipped_pipeline(store: Store):
    base = pool("630.306#a@2017-01-01", "630.310#b@2017-01-01", "300.101#a@2017-01-01")
    e = expander(store, budget=0).expand(base, final_k=2, as_of="2018-01-01")
    assert ids(e) == ["630.306#a@2017-01-01", "630.310#b@2017-01-01"]
    assert e.admitted == ()


def test_the_stage_output_is_the_whole_pool_and_final_is_the_cut(store: Store):
    base = pool("630.306#a@2017-01-01", "300.101#a@2017-01-01", "630.310#a@2017-01-01")
    e = expander(store, budget=1).expand(base, final_k=2, as_of="2018-01-01")
    assert len(e.final) == 2
    assert [c.version_id for c in e.ordered][:2] == ids(e)
    assert len(e.ordered) == 4          # three from hop 1, one admitted


def test_a_bad_budget_or_depth_is_refused():
    with Store(":memory:") as s:
        with pytest.raises(ValueError):
            ReferenceExpander(s, budget=-1)
        with pytest.raises(ValueError):
            ReferenceExpander(s, depth=0)


# -- depth and cycles ------------------------------------------------------------


def test_depth_two_does_not_follow_what_it_just_admitted(store: Store):
    e = expander(store, budget=4, depth=2).expand(pool("630.306#a@2017-01-01"), final_k=8,
                                                  as_of="2018-01-01")
    assert [a.chunk_id for a in e.admitted] == ["630.306#b"]
    assert e.depth_reached == 2


def test_depth_three_follows_the_reference_the_reference_made(store: Store):
    e = expander(store, budget=4, depth=3).expand(pool("630.306#a@2017-01-01"), final_k=8,
                                                  as_of="2018-01-01")
    assert [a.chunk_id for a in e.admitted] == ["630.306#b", "630.310#a"]
    assert [a.depth for a in e.admitted] == [2, 3]
    assert e.depth_reached == 3


def test_a_citation_cycle_terminates_and_admits_each_chunk_once(store: Store):
    """§630.306(a) -> (b) -> §630.310 -> §630.310(b) -> §630.306(a), which is already held.

    The depth cap is not what stops this. The visited set is: a target the evidence set
    already contains is not a dangling reference, so it is never followed, and the loop
    closes one hop after it opens however large the budget is.
    """
    e = expander(store, budget=16, depth=8).expand(pool("630.306#a@2017-01-01"), final_k=16,
                                                   as_of="2018-01-01")
    admitted = [a.chunk_id for a in e.admitted]
    assert admitted == sorted(set(admitted), key=admitted.index)    # no repeats
    assert "630.306#a" not in admitted                              # the loop's start
    assert len(ids(e)) == len(set(ids(e)))
    assert e.depth_reached <= 8


# -- predicates ------------------------------------------------------------------


def test_a_reference_is_followed_into_the_version_in_force_on_the_asked_date(store: Store):
    """A reference names a chunk id, not a version id; the as-of predicate picks the version."""
    old = expander(store, budget=1).expand(pool("630.310#b@2017-01-01"), final_k=2,
                                           as_of="2018-01-01")
    new = expander(store, budget=1).expand(pool("630.310#b@2017-01-01"), final_k=2,
                                           as_of="2021-01-01")
    assert [a.version_id for a in old.admitted] == ["630.306#a@2017-01-01"]
    assert [a.version_id for a in new.admitted] == ["630.306#a@2020-08-10"]


def test_a_superseded_version_is_never_admitted_beside_the_one_in_force(store: Store):
    e = expander(store, budget=4).expand(pool("630.310#b@2017-01-01"), final_k=8,
                                         as_of="2021-01-01")
    assert "630.306#a@2017-01-01" not in ids(e)


def test_a_reference_into_an_excluded_part_is_not_followed(store: Store):
    """§630.201(a) cites 5 CFR 531.404(a). Part 531 governs the GS and not the FWS."""
    inside = expander(store, budget=1).expand(pool("630.201#a@2017-01-01"), final_k=2,
                                              as_of="2018-01-01")
    outside = expander(store, budget=1).expand(pool("630.201#a@2017-01-01"), final_k=2,
                                               as_of="2018-01-01", exclude_parts=["531"])
    assert [a.chunk_id for a in inside.admitted] == ["531.404#a"]
    assert outside.admitted == ()
    assert outside.dangling == 0        # not "refused after ranking"; never a candidate


def test_an_excluded_reference_does_not_spend_a_slot(store: Store):
    """The refusal returns the budget to the first hop rather than shortening the answer."""
    base = pool("630.201#a@2017-01-01", "300.101#a@2017-01-01")
    e = expander(store, budget=1).expand(base, final_k=2, as_of="2018-01-01",
                                         exclude_parts=["531"])
    assert ids(e) == ["630.201#a@2017-01-01", "300.101#a@2017-01-01"]


def test_an_authority_ceiling_applies_to_the_second_hop_too(store: Store):
    """Same clause, same call site as ``Store.search``. Regulation is 2; a ceiling of 1
    admits statute only, so nothing in this corpus can be reached by a reference either."""
    e = expander(store, budget=2, max_authority=1).expand(
        pool("630.306#a@2017-01-01"), final_k=4, as_of="2018-01-01")
    assert e.admitted == ()


# -- the trace -------------------------------------------------------------------


def make(store: Store, **kw) -> MultiHopRetriever:
    kw.setdefault("final_k", 2)
    return MultiHopRetriever(store=store, candidates_lexical=50, candidates_dense=50,
                             rerank_top_k=20, parts_universe=PARTS, **kw)


def test_the_retriever_records_the_hop_as_its_own_stage(store: Store):
    trace = make(store, hop_budget=2).retrieve("restored annual leave scheduled",
                                               as_of="2018-01-01")
    admissions = getattr(trace, TRACE_ATTR)
    assert admissions, "nothing was admitted; the fixture no longer exercises the hop"
    assert all(a.reference and a.source for a in admissions)
    assert HOP_STAGE in trace.timings
    assert {a.version_id for a in admissions} <= set(trace.final)


def test_the_retriever_with_no_budget_matches_the_single_hop_retriever(store: Store):
    query = "restored annual leave scheduled and used"
    plain = Retriever(store=store, candidates_lexical=50, candidates_dense=50,
                      rerank_top_k=20, final_k=2, parts_universe=PARTS)
    assert (make(store, hop_budget=0).retrieve(query, as_of="2018-01-01").final
            == plain.retrieve(query, as_of="2018-01-01").final)


def test_the_retriever_carries_the_askers_scope_into_the_second_hop(store: Store):
    query = "accrual of annual leave computed"
    fws = make(store, hop_budget=2).retrieve(query, as_of="2018-01-01",
                                             scope=Scope.of(pay_system="FWS"))
    wide = make(store, hop_budget=2).retrieve(query, as_of="2018-01-01",
                                              scope=GOVERNMENT_WIDE)
    assert not any(v.startswith("531.") for v in fws.final)
    assert any(v.startswith("531.") for v in wide.final)


def test_the_hop_is_counted_in_the_total_timing(store: Store):
    trace = make(store, hop_budget=2).retrieve("restored annual leave", as_of="2018-01-01")
    assert trace.timings["total"] >= trace.timings[HOP_STAGE] > 0.0
