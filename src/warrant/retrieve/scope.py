"""Which regulation governs this person.

The other half of "applicability-aware temporal RAG", and the half that is easy to overclaim.
To be explicit, once, in code as well as in ARCHITECTURE.md section 3: **this is not access
control.** eCFR is published law, nothing here is confidential, and nothing can leak. Citing
a rule that does not govern the asker is a correctness failure, not a security breach, and
the metric is an applicability error rate.

The restrictions below are read off the parts' own titles, not invented:

    511  Classification under the General Schedule      -> GS
    531  Pay under the General Schedule                 -> GS
    532  Prevailing Rate Systems                        -> FWS
    534  Pay under Other Systems                        -> SES and other systems
    317  Employment in the Senior Executive Service     -> SES
    315  Career and Career-Conditional Employment       -> competitive service
    316  Temporary and Term Employment                  -> competitive service
    337  Examining System                               -> competitive service

Every other part in the corpus is government-wide and applies to everyone. That asymmetry is
deliberate: the default is *applies*, and a restriction has to be justified by the source.
Inventing restrictions would make the system confidently refuse to answer questions it can
answer, which is a worse failure than being over-broad.

The predicate is pushed into the retrieval query rather than applied to results. The
justification here is correctness and cost, not confidentiality: text that cannot govern the
asker should never consume a candidate slot or reranker budget. That the wasted slots are
real and measurable is shown by the temporal ablation, where superseded near-duplicates
crowd out correct evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: part -> facet -> the facet values that part governs.
PART_RESTRICTIONS: dict[str, dict[str, frozenset[str]]] = {
    "511": {"pay_system": frozenset({"GS"})},
    "531": {"pay_system": frozenset({"GS"})},
    "532": {"pay_system": frozenset({"FWS"})},
    "534": {"pay_system": frozenset({"SES", "other"})},
    "317": {"pay_system": frozenset({"SES"})},
    "315": {"service": frozenset({"competitive"})},
    "316": {"service": frozenset({"competitive"})},
    "337": {"service": frozenset({"competitive"})},
}

#: Facets a profile may declare. Anything else is ignored rather than silently mismatched.
KNOWN_FACETS = frozenset({"pay_system", "service"})


def known_values(facet: str) -> frozenset[str]:
    """Every value of ``facet`` that some part actually governs."""
    values: set[str] = set()
    for restriction in PART_RESTRICTIONS.values():
        values |= restriction.get(facet, frozenset())
    return frozenset(values)


@dataclass(frozen=True)
class Scope:
    """Who is asking. An unspecified facet is not filtered on.

    That default is the honest one: a profile that says nothing about pay system is a
    government-wide view, not an assertion that no pay-system rules apply. Treating unknown
    as restrictive would hide correct answers behind missing metadata.
    """

    facets: dict[str, str] = field(default_factory=dict)

    @classmethod
    def of(cls, **facets: str) -> Scope:
        """Build a profile, rejecting both unknown facet names and unknown facet values.

        Validating the *value* matters more than validating the name. An unknown name is
        loud; an unknown value used to be silent and dangerous: ``pay_system="bogus"``
        matches no part's allowed set, so ``governs`` returned False for every restricted
        part and quietly removed five parts -- 41% of the corpus -- before returning a
        confident, degraded answer with HTTP 200. A typo has to be an error, not a filter.
        """
        unknown = set(facets) - KNOWN_FACETS
        if unknown:
            raise ValueError(
                f"unknown scope facet(s): {sorted(unknown)}; "
                f"known facets are {sorted(KNOWN_FACETS)}")
        clean = {k: v for k, v in facets.items() if v}
        for facet, value in clean.items():
            allowed = known_values(facet)
            if value not in allowed:
                raise ValueError(
                    f"unknown {facet} {value!r}; known values are {sorted(allowed)}")
        return cls(facets=clean)

    def governs(self, part: str) -> bool:
        """Does the regulation in ``part`` apply to this profile?"""
        for facet, allowed in PART_RESTRICTIONS.get(part, {}).items():
            declared = self.facets.get(facet)
            if declared is not None and declared not in allowed:
                return False
        return True

    def excluded_parts(self, universe: list[str]) -> list[str]:
        """Parts to keep out of the retrieval query entirely."""
        return sorted(p for p in universe if not self.governs(p))

    def describe(self) -> str:
        if not self.facets:
            return "government-wide"
        return ", ".join(f"{k}={v}" for k, v in sorted(self.facets.items()))


#: The unrestricted view: every part applies. Used when a question carries no profile.
GOVERNMENT_WIDE = Scope()
