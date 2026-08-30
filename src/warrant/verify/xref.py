"""In-text cross-references, and the ones the evidence set does not cover.

A cited paragraph rarely stands alone. Measured over the 9,961 in-force chunks: 15.7% carry a
``§ 630.309``, 9.6% a ``5 U.S.C. 6304(d)``, 8.3% a ``paragraph (b) of this section``, and
6.3% a bare ``this subpart``. When retrieval hands the generator ``630.306#a`` and not
``630.310(d)``, the answer is built on one link of a conditional chain and every span
alignment still passes -- the cited text does say what the claim says, it just is not the
whole rule.

So the reference is parsed and resolved to a chunk id, and a reference whose target is absent
from the evidence set is reported as a ``DanglingReference``. Three outcomes, kept apart
because they mean different things and have different fixes:

    missing    the target is in the corpus and was not retrieved   -- a retrieval gap
    outside    the target is not a 5 CFR chapter I chunk at all    -- a corpus-scope limit
    unscoped   "this subpart" names no single chunk                -- not actionable

Collapsing them would let the corpus boundary (every U.S.C. reference is unresolvable here by
construction) inflate a number that is supposed to be about retrieval.

Targets are **chunk ids**, not version ids: a reference in the 2017 text of §630.306 names
§630.310, not any particular version of it. Choosing the version is the as-of predicate's job
and it already has one.
"""

from __future__ import annotations

import re
from collections.abc import Container, Iterable, Mapping
from dataclasses import dataclass

#: One paragraph designator as the CFR writes it, matching the character class
#: ``corpus.parse`` accepts when it builds anchors -- (a), (12), (iv), (B).
_DES = r"\(\s*[A-Za-z0-9]{1,4}\s*\)"
#: A run of them: (b)(2), (a)(3)(v).
_DES_RUN = rf"(?:{_DES}\s*)+"
#: Section numbers in 5 CFR chapter I are part.number: 630.306, 890.1603, 532.285.
_SEC = r"\d{3}\.\d{1,4}"
#: What separates two references written as one phrase: "§§ 316.701 and 316.702",
#: "paragraphs (a), (c), and (d)". Ordinary sentence commas are excluded by requiring
#: another reference token to follow.
_JOIN = r"(?:\s*(?:,|;)?\s*(?:and|or|through|to)\s+|\s*,\s*)"

_PARAGRAPH_REF = re.compile(
    rf"\bparagraphs?\s+(?P<groups>{_DES_RUN}(?:{_JOIN}{_DES_RUN})*)"
    rf"(?:\s+of\s+(?:(?P<this>this\s+section)|§\s*(?P<sec>{_SEC})))?",
    re.IGNORECASE,
)
_SECTION_REF = re.compile(
    rf"§{{1,2}}\s*(?P<first>{_SEC})(?P<fdes>{_DES_RUN})?"
    rf"(?P<more>(?:{_JOIN}(?:§{{1,2}}\s*)?{_SEC}(?:{_DES_RUN})?)*)"
)
_USC_REF = re.compile(
    r"(?P<title>\d{1,2})\s+U\.S\.C\.\s*(?:App\.\s*)?"
    rf"(?P<lead>(?:chapter|subchapter|section|part)s?\s+)?(?P<num>\d{{1,5}}[A-Za-z]?)"
    rf"(?P<des>{_DES_RUN})?"
)
_CFR_REF = re.compile(
    rf"(?P<title>\d{{1,2}})\s+CFR\s+(?:parts?\s+)?(?P<num>\d{{2,3}}(?:\.\d{{1,4}})?)"
    rf"(?P<des>{_DES_RUN})?"
)
_SCOPE_REF = re.compile(
    r"\bthis\s+(?:section|subpart|part|chapter|title)\b"
    r"|\bsubparts?\s+[A-Z]{1,2}\b(?:\s+of\s+(?:this\s+part|part\s+\d{3}))?"
    r"|\bparts?\s+\d{3}(?:\s*(?:,|;)?\s*(?:and|or)?\s*\d{3})*\b",
    re.IGNORECASE,
)

#: Priority order, and it is load-bearing. "paragraph (b) of § 630.310" must be read as one
#: paragraph reference; matching sections first would consume the "§ 630.310" and leave the
#: paragraph pointing at the citing section instead. Likewise "this section" inside a
#: paragraph reference is part of that reference, not a scope reference of its own.
_PATTERNS = (
    ("paragraph", _PARAGRAPH_REF),
    ("section", _SECTION_REF),
    ("usc", _USC_REF),
    ("cfr", _CFR_REF),
    ("scope", _SCOPE_REF),
)

#: The corpus is 5 CFR chapter I. A reference into another title is nameable and outside.
CORPUS_TITLE = 5


@dataclass(frozen=True)
class Reference:
    """One reference phrase and the chunk ids it names.

    ``targets`` is a tuple because a single phrase routinely names several: "paragraphs (b)(1)
    and (b)(2) of this section" is one reference with two targets, and splitting it into two
    ``Reference`` objects would make the spans overlap and the text unreadable.

    A section-level target has no ``#anchor``. It is satisfied by *any* paragraph of that
    section being present, which is the right reading of "as provided in § 351.703".
    """

    kind: str                       # paragraph | section | usc | cfr | scope
    text: str
    span: tuple[int, int]
    targets: tuple[str, ...] = ()

    @property
    def resolvable(self) -> bool:
        """Does this reference name something a 5 CFR chapter I evidence set could contain?"""
        return self.kind in ("paragraph", "section") and bool(self.targets)


@dataclass(frozen=True)
class DanglingReference:
    """A cited chunk points somewhere the evidence set does not go."""

    source: str                     # version id of the citing chunk
    target: str                     # chunk id, or the printed phrase for outside/unscoped
    status: str                     # missing | outside | unscoped
    reference: Reference | None = None


def chunk_id_of(version_id: str) -> str:
    """``630.306#a@2017-01-01`` -> ``630.306#a``. Passes a bare chunk id through."""
    return version_id.split("@", 1)[0]


def section_of(chunk_id: str) -> str:
    return chunk_id.split("#", 1)[0]


def _anchor(designators: str) -> str:
    """``(b)(2)`` -> ``b-2``, the anchor form ``corpus.parse`` writes into the store."""
    return "-".join(m.strip() for m in re.findall(r"\(\s*([A-Za-z0-9]{1,4})\s*\)", designators))


def _split_designator_runs(groups: str) -> list[str]:
    """The designator runs in "paragraphs (b)(1) and (b)(2)", as anchors.

    Split on the joiners rather than on every ``)(``: "(b)(1)" is one two-level address and
    "(b), (c)" is two one-level ones, and only the separator tells them apart.
    """
    runs = [r for r in re.split(_JOIN, groups) if r.strip()]
    return [a for a in (_anchor(r) for r in runs) if a]


def _paragraph_targets(m: re.Match[str], section_id: str) -> tuple[str, ...]:
    target_section = m.group("sec") or section_id
    if not target_section:
        return ()
    return tuple(f"{target_section}#{a}" for a in _split_designator_runs(m.group("groups")))


def _section_targets(m: re.Match[str]) -> tuple[str, ...]:
    out: list[str] = []
    first, fdes = m.group("first"), m.group("fdes") or ""
    out.append(f"{first}#{_anchor(fdes)}" if fdes.strip() else first)
    for sec, des in re.findall(rf"({_SEC})({_DES_RUN})?", m.group("more") or ""):
        out.append(f"{sec}#{_anchor(des)}" if des and des.strip() else sec)
    return tuple(dict.fromkeys(out))


def find_references(text: str, *, section_id: str = "") -> list[Reference]:
    """Every reference phrase in ``text``, in document order, non-overlapping.

    ``section_id`` is the citing chunk's section; without it "paragraph (b) of this section"
    is still found but cannot be resolved, and comes back with no targets rather than with a
    guessed one.
    """
    taken: list[tuple[int, int]] = []
    found: list[Reference] = []
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            start, end = m.span()
            if any(start < b and a < end for a, b in taken):
                continue
            taken.append((start, end))
            if kind == "paragraph":
                targets = _paragraph_targets(m, section_id)
            elif kind == "section":
                targets = _section_targets(m)
            elif kind in ("usc", "cfr"):
                # Nameable, but only a same-title CFR reference can be a chunk here.
                title = int(m.group("title"))
                num = m.group("num")
                des = (m.group("des") or "").strip()
                if kind == "cfr" and title == CORPUS_TITLE and "." in num:
                    targets = (f"{num}#{_anchor(des)}" if des else num,)
                else:
                    targets = ()
            else:
                targets = ()
            found.append(Reference(kind=kind, text=m.group(0).strip(),
                                   span=(start, end), targets=targets))
    return sorted(found, key=lambda r: r.span)


def enumerated_children(chunk_id: str, in_corpus: Iterable[str]) -> list[str]:
    """The immediate sub-paragraphs of ``chunk_id``: ``630.306#a`` -> ``630.306#a-1``, ``-2``.

    Immediate only. ``a-1-i`` hangs off ``a-1``, and pulling every descendant of a chapeau
    into the evidence set would drag whole sections back in through a check whose point is to
    name the one paragraph that is missing.
    """
    prefix = f"{chunk_id}-"
    return sorted(c for c in in_corpus
                  if c.startswith(prefix) and "-" not in c[len(prefix):])


def dangling_references(evidence: Mapping[str, str], *, in_corpus: Container[str] = (),
                        include_outside: bool = False) -> list[DanglingReference]:
    """References made by the cited chunks that the evidence set does not satisfy.

    ``evidence`` maps version id to chunk text -- the same shape ``generate.answer.Answer``
    already carries as ``cited``, so nothing upstream has to be reshaped to call this.

    A section-level target is satisfied by any evidence chunk from that section. A reference a
    chunk makes to itself is not a reference to anywhere else and is dropped; so is a
    reference to a chunk id the corpus does not contain, which is a resolution failure of this
    module rather than a retrieval failure, and is reported as ``outside`` only when asked
    for.
    """
    present_chunks = {chunk_id_of(v) for v in evidence}
    present_sections = {section_of(c) for c in present_chunks}
    out: list[DanglingReference] = []
    for version_id, text in sorted(evidence.items()):
        source_chunk = chunk_id_of(version_id)
        for ref in find_references(text, section_id=section_of(source_chunk)):
            if not ref.resolvable:
                if include_outside and ref.kind in ("usc", "cfr", "scope"):
                    out.append(DanglingReference(
                        source=version_id, target=ref.text,
                        status="unscoped" if ref.kind == "scope" else "outside",
                        reference=ref))
                continue
            for target in ref.targets:
                if target == source_chunk:
                    continue
                if "#" in target:
                    if target in present_chunks:
                        continue
                    status = "missing" if target in in_corpus else "outside"
                else:
                    if target in present_sections:
                        continue
                    status = "missing" if any(
                        section_of(c) == target for c in in_corpus) else "outside"
                if status == "outside" and not include_outside:
                    continue
                out.append(DanglingReference(source=version_id, target=target,
                                             status=status, reference=ref))
    return out


def corpus_chunk_ids(store: object, *, as_of: str, system_time: str | None = None) -> set[str]:
    """Every chunk id in force on ``as_of``. The membership set ``in_corpus`` wants.

    Chunk ids rather than version ids: see the module docstring. Typed loosely so this module
    does not import the store, which keeps the tests store-free.
    """
    rows = store.as_of(as_of, system_time=system_time)  # type: ignore[attr-defined]
    return {r["chunk_id"] for r in rows}
