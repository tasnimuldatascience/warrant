"""The applicability model.

The default matters more than the restrictions: a part with no recorded restriction governs
everyone. Inventing restrictions would make the system confidently refuse questions it can
answer, which is worse than being over-broad, and it would be invisible in any metric that
only counts wrong answers.
"""

from __future__ import annotations

import pytest

from warrant.retrieve.scope import GOVERNMENT_WIDE, PART_RESTRICTIONS, Scope

UNIVERSE = ["300", "315", "317", "511", "531", "532", "534", "550", "630", "890"]


def test_unrestricted_parts_govern_everyone():
    for scope in (GOVERNMENT_WIDE, Scope.of(pay_system="GS"), Scope.of(pay_system="FWS")):
        assert scope.governs("630"), "leave rules are government-wide"
        assert scope.governs("550")
        assert scope.governs("890")


def test_pay_system_restrictions_follow_the_part_titles():
    """531 is 'Pay under the General Schedule', 532 is 'Prevailing Rate Systems'. The
    restriction is read off the source, not invented."""
    gs, fws = Scope.of(pay_system="GS"), Scope.of(pay_system="FWS")
    assert gs.governs("531") and not gs.governs("532")
    assert fws.governs("532") and not fws.governs("531")


def test_ses_parts_exclude_general_schedule_employees():
    ses, gs = Scope.of(pay_system="SES"), Scope.of(pay_system="GS")
    assert ses.governs("317") and not gs.governs("317")


def test_unspecified_facet_is_not_filtered_on():
    """A profile silent about pay system is a government-wide view, not an assertion that no
    pay rules apply. Treating unknown as restrictive would hide correct answers behind
    missing metadata."""
    assert GOVERNMENT_WIDE.governs("531")
    assert GOVERNMENT_WIDE.governs("532")
    assert GOVERNMENT_WIDE.excluded_parts(UNIVERSE) == []


def test_facets_are_independent():
    """A service-level profile must not exclude pay parts it says nothing about."""
    competitive = Scope.of(service="competitive")
    assert competitive.governs("531") and competitive.governs("532")
    assert competitive.governs("315")


def test_excluded_parts_lists_only_what_is_ruled_out():
    assert Scope.of(pay_system="GS").excluded_parts(UNIVERSE) == ["317", "532", "534"]
    assert Scope.of(pay_system="FWS").excluded_parts(UNIVERSE) == ["317", "511", "531", "534"]


def test_unknown_facet_is_rejected_rather_than_ignored():
    """A silently ignored facet would produce a profile that quietly filters nothing."""
    with pytest.raises(ValueError, match="unknown scope facet"):
        Scope.of(clearance="secret")


def test_empty_facet_value_is_dropped():
    assert Scope.of(pay_system="").facets == {}


def test_describe_is_stable_and_readable():
    assert GOVERNMENT_WIDE.describe() == "government-wide"
    assert Scope.of(pay_system="GS", service="competitive").describe() == (
        "pay_system=GS, service=competitive")


def test_every_restriction_names_a_known_facet():
    """A typo in a facet name would make the restriction silently unenforceable."""
    from warrant.retrieve.scope import KNOWN_FACETS

    for part, restriction in PART_RESTRICTIONS.items():
        assert set(restriction) <= KNOWN_FACETS, part
        for values in restriction.values():
            assert values, f"{part} restricts a facet to nothing"
