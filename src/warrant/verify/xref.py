"""In-text cross-references, and the ones the evidence set does not cover.

A cited paragraph rarely stands alone. **43.5% of the 9,961 in-force chunks make at least one
reference**: 15.5% a ``§ 630.309``, 11.3% a ``paragraph (b) of this section``, 10.3% a
``5 U.S.C. 6304(d)``, 3.2% a ``5 CFR part 550``, and 19.3% a scope phrase such as ``this
subpart``. When retrieval hands the generator ``630.306#a`` and not ``630.310(d)``, the
answer is built on one link of a conditional chain and every span alignment still passes --
the cited text does say what the claim says, it just is not the whole rule.

So the reference is parsed and resolved to a chunk id, and a reference whose target is absent
from the evidence set is reported as a ``DanglingReference``. Three outcomes, kept apart
because they mean different things and have different fixes:

    missing    the target is in the corpus and was not retrieved   -- a retrieval gap
    outside    the target is not a 5 CFR chapter I chunk at all    -- a corpus-scope limit
    unscoped   "this subpart" names no single chunk                -- not actionable

Collapsing them would let the corpus boundary (every U.S.C. reference is unresolvable here by
construction) inflate a number that is supposed to be about retrieval.

Resolution is measured, not assumed: of the 4,281 targets this module emits for sections the
corpus actually holds, 94.7% resolve to that exact chunk id, 4.6% to an ancestor and 0.8%
only to the section. The residue is not a parsing failure here -- it is the chunker running
a paragraph's first item together with its chapeau, so ``300.201#a-1`` is text inside
``300.201#a`` and no such id is ever written.

Targets are **chunk ids**, not version ids: a reference in the 2017 text of §630.306 names
§630.310, not any particular version of it. Choosing the version is the as-of predicate's job
and it already has one.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass

#: One paragraph designator as the CFR writes it, matching the character class
#: ``corpus.parse`` accepts when it builds anchors -- (a), (12), (iv), (B).
_DES = r"\(\s*[A-Za-z0-9]{1,4}\s*\)"
#: A run of them: (b)(2), (a)(3)(v).
_DES_RUN = rf"(?:{_DES}\s*)+"
#: Section numbers. Every part in this corpus is three digits, but the Civil Service Rules
#: are cited as ``§ 6.7 of this chapter`` from five sections of it, and a pattern that cannot
#: match them reports nothing rather than reporting ``outside`` -- which is the difference
#: between a corpus boundary that is visible in the numbers and one that is not.
_SEC = r"\d{1,3}\.\d{1,4}"
#: What separates two references written as one phrase: "§§ 316.701 and 316.702",
#: "paragraphs (a), (c), and (d)". Ordinary sentence commas are excluded by requiring
#: another reference token to follow.
_JOIN = r"(?:\s*(?:,|;)?\s*(?:and|or|through|to)\s+|\s*,\s*)"

_PARAGRAPH_REF = re.compile(
    rf"\bparagraphs?\s+(?P<groups>{_DES_RUN}(?:{_JOIN}{_DES_RUN})*)"
    # ``\s*of`` and not ``\s+of``: ``_DES_RUN`` already consumes the space after "(b)", so
    # requiring another one silently detached every "of this section" and "of § 630.310" from
    # the paragraph it belongs to -- which left the reference resolving against the citing
    # section and the trailing "this section" reappearing as a scope reference of its own.
    rf"(?:\s*of\s+(?:(?P<this>this\s+section)|§\s*(?P<sec>{_SEC})))?",
    re.IGNORECASE,
)
_SECTION_REF = re.compile(
    rf"§{{1,2}}\s*(?P<first>{_SEC})(?P<fdes>{_DES_RUN})?"
    rf"(?P<more>(?:{_JOIN}(?:§{{1,2}}\s*)?{_SEC}(?:{_DES_RUN})?)*)"
)
#: ``re.IGNORECASE`` because the corpus writes "5 U.S.C. Chapter 43" and "5 U.S.C.
#: Section 7116(a)(7)" as often as the lowercase form, and without the flag those 13 chunks
#: were silently not references at all.
_USC_REF = re.compile(
    r"(?P<title>\d{1,2})\s+U\.S\.C\.\s*(?:App\.\s*)?"
    rf"(?P<lead>(?:chapter|subchapter|section|part)s?\s+)?(?P<num>\d{{1,5}}[A-Za-z]?)"
    rf"(?P<des>{_DES_RUN})?",
    re.IGNORECASE,
)
_CFR_REF = re.compile(
    rf"(?P<title>\d{{1,2}})\s+CFR\s+(?:parts?\s+)?(?P<num>\d{{2,3}}(?:\.\d{{1,4}})?)"
    rf"(?P<des>{_DES_RUN})?"
)
_SCOPE_REF = re.compile(
    r"\b(?P<this_section>this\s+section)\b"
    r"|\bthis\s+(?:subpart|part|chapter|title)\b"
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
        return bool(self.targets)


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


def anchor_of(chunk_id: str) -> str:
    return chunk_id.partition("#")[2]


def _levels(designators: str) -> tuple[str, ...]:
    return tuple(m.strip()
                 for m in re.findall(r"\(\s*([A-Za-z0-9]{1,4})\s*\)", designators))


def _anchor(designators: str) -> str:
    """``(b)(2)`` -> ``b-2``, the anchor form ``corpus.parse`` writes into the store."""
    return "-".join(_levels(designators))


_RANGE_JOIN = re.compile(r"\b(?:through|to)\b", re.IGNORECASE)
_ROMAN = re.compile(r"[ivxlcdm]+", re.IGNORECASE)
_ROMAN_VALUE = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
#: A range wider than this is a misread of the numbering system, not a citation. "(c)
#: through (d)" parses as roman 100-500 if roman is tried first, and the cap is what stops
#: 400 fabricated targets from reaching the dangling-reference count.
MAX_RANGE = 24


def _roman(token: str) -> int | None:
    if not _ROMAN.fullmatch(token):
        return None
    low, total = token.lower(), 0
    for i, ch in enumerate(low):
        v = _ROMAN_VALUE[ch]
        nxt = _ROMAN_VALUE.get(low[i + 1]) if i + 1 < len(low) else None
        total += -v if nxt and nxt > v else v
    return total


def _to_roman(n: int, upper: bool) -> str:
    out, rest = "", n
    for value, sym in ((1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"),
                       (90, "xc"), (50, "l"), (40, "xl"), (10, "x"), (9, "ix"),
                       (5, "v"), (4, "iv"), (1, "i")):
        while rest >= value:
            out += sym
            rest -= value
    return out.upper() if upper else out


def _alpha(token: str) -> int | None:
    if not token.isalpha():
        return None
    n = 0
    for ch in token.lower():
        n = n * 26 + (ord(ch) - 96)
    return n


def _to_alpha(n: int, upper: bool) -> str:
    out = ""
    while n:
        n, r = divmod(n - 1, 26)
        out = chr(97 + r) + out
    return out.upper() if upper else out


def _between(lo: str, hi: str) -> list[str]:
    """The designators strictly between ``lo`` and ``hi``, or [] if they are not a range.

    Digits, then roman, then letters -- except that two single characters are read as letters
    first. "(c) through (e)" is (c)(d)(e) and not the roman 100-through-500 it also parses as;
    "(i) through (iv)" is roman because "iv" cannot be anything else.
    """
    if lo.isdigit() and hi.isdigit():
        return [str(n) for n in range(int(lo) + 1, int(hi))]
    orders: list[tuple] = []
    single = len(lo) == 1 and len(hi) == 1
    roman = (_roman(lo), _roman(hi), _to_roman)
    alpha = (_alpha(lo), _alpha(hi), _to_alpha)
    orders = [alpha, roman] if single else [roman, alpha]
    for a, b, render in orders:
        if a is None or b is None or not 0 < b - a <= MAX_RANGE:
            continue
        return [render(n, lo.isupper()) for n in range(a + 1, b)]
    return []


def _designator_runs(groups: str) -> list[str]:
    """The anchors named by "paragraphs (b)(1) and (b)(2)", "(a)(1) through (4)", "(a) or (c)".

    Two things the CFR does that a naive split on the joiners gets wrong, both of which
    manufacture targets that exist nowhere and so are reported as dangling:

    **Elided prefixes.** "paragraphs (a)(1) through (4)" names ``a-1`` .. ``a-4``, not ``a-1``
    and ``4``. A run shorter than the one before it continues that one; 351.403 and 536.308
    are among the sections where the short form produced a target with no section-level
    designator at all.

    **Ranges.** "through" and "to" name the endpoints *and* everything between them. Emitting
    only the endpoints leaves the interior paragraphs unchecked, which is the direction that
    understates the problem this module measures.
    """
    runs = list(re.finditer(_DES_RUN, groups))
    out: list[tuple[str, ...]] = []
    previous: tuple[str, ...] | None = None
    for i, m in enumerate(runs):
        current = _levels(m.group(0))
        if not current:
            continue
        if previous and len(current) < len(previous):
            current = previous[:len(previous) - len(current)] + current
        gap = groups[runs[i - 1].end():m.start()] if i else ""
        if previous and _RANGE_JOIN.search(gap) and current[:-1] == previous[:-1]:
            out.extend(previous[:-1] + (d,) for d in _between(previous[-1], current[-1]))
        out.append(current)
        previous = current
    return list(dict.fromkeys("-".join(r) for r in out))


_PLAIN_LEVEL = re.compile(r"[A-Za-z]{1,4}|\d{1,3}")


def _relative_prefix(anchor: str) -> str:
    """The address a bare "paragraphs (1) through (3)" hangs off, or "".

    Top-level CFR paragraphs are lettered, so a bare reference opening with a *digit* is never
    to (1) of the section -- it is to a sibling inside the paragraph doing the citing.
    630.201(b)(6) says "paragraphs (2) through (5)" and means (b)(2) through (b)(5); resolved
    flat it names ``630.201#2``, which exists nowhere and was charged to the corpus boundary.

    Only the lettered head of the citing anchor is kept, and only when every level of it is a
    plain designator: ``p4`` is the address ``corpus.parse`` gives a paragraph with no
    designator at all, and prefixing with it would invent an address rather than recover one.
    """
    levels = anchor.split("-") if anchor else []
    head: list[str] = []
    for level in levels:
        if not _PLAIN_LEVEL.fullmatch(level):
            return ""
        if level.isdigit():
            break
        head.append(level)
    return "-".join(head)


def _paragraph_targets(m: re.Match[str], section_id: str, anchor: str) -> tuple[str, ...]:
    target_section = m.group("sec") or section_id
    if not target_section:
        return ()
    runs = _designator_runs(m.group("groups"))
    bare = not m.group("sec") and not m.group("this")
    if bare and anchor and runs and runs[0][0].isdigit():
        prefix = _relative_prefix(anchor)
        if prefix:
            runs = [f"{prefix}-{r}" for r in runs]
    return tuple(f"{target_section}#{a}" for a in runs)


def _section_targets(m: re.Match[str]) -> tuple[str, ...]:
    out: list[str] = []
    first, fdes = m.group("first"), m.group("fdes") or ""
    out.append(f"{first}#{_anchor(fdes)}" if fdes.strip() else first)
    for sec, des in re.findall(rf"({_SEC})({_DES_RUN})?", m.group("more") or ""):
        out.append(f"{sec}#{_anchor(des)}" if des and des.strip() else sec)
    return tuple(dict.fromkeys(out))


def find_references(text: str, *, section_id: str = "",
                    anchor: str = "") -> list[Reference]:
    """Every reference phrase in ``text``, in document order, non-overlapping.

    ``section_id`` is the citing chunk's section; without it "paragraph (b) of this section"
    is still found but cannot be resolved, and comes back with no targets rather than with a
    guessed one. ``anchor`` is the citing paragraph's own address, which only a bare
    digit-leading reference needs -- see ``_relative_prefix``.
    """
    # Each pattern runs over what the higher-priority ones left, with their matches blanked
    # to a character no pattern can match. Discarding a lower-priority match wholesale
    # instead -- which is what an overlap test does -- loses the part that did not overlap:
    # in "paragraph (d) of § 630.309 and § 630.310(a)" the section pattern matches the two
    # section numbers as one phrase, the first of which the paragraph reference already owns,
    # and § 630.310 disappeared with it.
    remaining = text
    found: list[Reference] = []
    for kind, pattern in _PATTERNS:
        masked = list(remaining)
        for m in pattern.finditer(remaining):
            start, end = m.span()
            masked[start:end] = "\x00" * (end - start)
            if kind == "paragraph":
                targets = _paragraph_targets(m, section_id, anchor)
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
            elif m.groupdict().get("this_section") and section_id:
                # "under this section" names the citing section, which the evidence set
                # already contains by construction. Left unresolved it was the single
                # largest entry in the unscoped column -- 28% of chunks -- and none of it
                # was ever actionable.
                targets = (section_id,)
            else:
                targets = ()
            found.append(Reference(kind=kind, text=m.group(0).strip(),
                                   span=(start, end), targets=targets))
        remaining = "".join(masked)
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


def nameable_ids(in_corpus: Iterable[str]) -> frozenset[str]:
    """Every chunk id the corpus contains, plus the ancestors and sections those imply.

    A reference names an address in the regulation; the store holds an address per *parsed*
    paragraph, and the two do not coincide. 890.102 is written with a paragraph (j) whose
    chapeau runs inline with (j)(1), so the parser emits ``j-1`` .. ``j-5`` and never a bare
    ``j`` -- and "paragraph (j)" then resolves to a chunk id present nowhere in the store.
    That accounted for most of the 11.9% of in-corpus targets that failed to resolve before
    this set existed, and every one of them was being charged to the corpus boundary as
    ``outside``, which is the column that is supposed to mean "not 5 CFR chapter I".

    Ancestors are materialised once rather than tested by prefix scan per target: the corpus
    is 9,961 chunks and the benchmark asks about thousands of targets.
    """
    out: set[str] = set()
    for chunk_id in in_corpus:
        out.add(chunk_id)
        section, _, anchor = chunk_id.partition("#")
        out.add(section)
        parts = anchor.split("-")
        for i in range(1, len(parts)):
            out.add(f"{section}#{'-'.join(parts[:i])}")
    return frozenset(out)


def resolve(target: str, nameable: Collection[str]) -> str:
    """``target`` as an address the corpus actually holds, or "" if it holds none.

    Walks up. 300.201 is written "(a) ... The Office does not release the following: (1) ...",
    so the whole of (a)(1) is inside the chunk addressed ``300.201#a`` and no ``#a-1`` is ever
    emitted; a reference to (a)(1) is answered by (a), and calling it a dangling reference
    would report a retrieval gap where the text is in fact right there. 4.7% of in-corpus
    targets resolve only through this fallback, and every one of them is a paragraph whose
    first item ``corpus.parse`` ran together with its chapeau.
    """
    if target in nameable:
        return target
    section, _, anchor = target.partition("#")
    levels = anchor.split("-") if anchor else []
    for i in range(len(levels) - 1, 0, -1):
        candidate = f"{section}#{'-'.join(levels[:i])}"
        if candidate in nameable:
            return candidate
    return section if section in nameable else ""


def _satisfies(target: str, present: Collection[str]) -> bool:
    """Is ``target`` covered by the chunk ids in ``present``?

    A section-level target is covered by any paragraph of that section -- the right reading of
    "as provided in § 351.703". A paragraph-level target is covered by itself or by its
    descendants, because "paragraph (j)" is answered by (j)(1) through (j)(5); it is *not*
    covered by its ancestors, since having the chapeau of (b) says nothing about what (b)(2)
    requires.
    """
    if "#" not in target:
        return any(section_of(c) == target for c in present)
    prefix = f"{target}-"
    return any(c == target or c.startswith(prefix) for c in present)


def dangling_references(evidence: Mapping[str, str], *, in_corpus: Iterable[str] = (),
                        include_outside: bool = False) -> list[DanglingReference]:
    """References made by the cited chunks that the evidence set does not satisfy.

    ``evidence`` maps version id to chunk text -- the same shape ``generate.answer.Answer``
    already carries as ``cited``, so nothing upstream has to be reshaped to call this.

    A reference a chunk makes to itself is not a reference to anywhere else and is dropped; so
    is a reference to an address the corpus does not contain, which is a corpus-scope limit
    rather than a retrieval failure, and is reported as ``outside`` only when asked for.
    """
    nameable = in_corpus if isinstance(in_corpus, frozenset) else nameable_ids(in_corpus)
    present = {chunk_id_of(v) for v in evidence}
    out: list[DanglingReference] = []
    for version_id, text in sorted(evidence.items()):
        source_chunk = chunk_id_of(version_id)
        for ref in find_references(text, section_id=section_of(source_chunk),
                                   anchor=anchor_of(source_chunk)):
            if not ref.resolvable:
                if include_outside and ref.kind in ("usc", "cfr", "scope"):
                    out.append(DanglingReference(
                        source=version_id, target=ref.text,
                        status="unscoped" if ref.kind == "scope" else "outside",
                        reference=ref))
                continue
            for target in ref.targets:
                located = resolve(target, nameable)
                if located == source_chunk or _satisfies(located or target, present):
                    continue
                status = "missing" if located else "outside"
                if status == "outside" and not include_outside:
                    continue
                out.append(DanglingReference(source=version_id, target=located or target,
                                             status=status, reference=ref))
    return out


def corpus_chunk_ids(store: object, *, as_of: str, system_time: str | None = None) -> set[str]:
    """Every chunk id in force on ``as_of``. The membership set ``in_corpus`` wants.

    Chunk ids rather than version ids: see the module docstring. Typed loosely so this module
    does not import the store, which keeps the tests store-free.
    """
    rows = store.as_of(as_of, system_time=system_time)  # type: ignore[attr-defined]
    return {r["chunk_id"] for r in rows}
