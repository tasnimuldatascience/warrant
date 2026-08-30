"""eCFR XML into the units Warrant retrieves and cites.

The regulation's own hierarchy is the chunking strategy. Title -> chapter -> part ->
subpart -> section -> paragraph is not an arbitrary segmentation someone tuned; it is how
the document is written, cross-referenced and amended. Inventing a fixed-token window on
top of it would throw away the one structure that makes citation and applicability
reasoning possible.

  section    the retrieval unit    <DIV8 TYPE="SECTION" N="630.1203">
  paragraph  the citation unit     <P>(a) An employee shall be entitled to ...

A citation therefore reads ``630.1203#a`` -- addressable, stable across snapshots when the
text is only amended, and exactly what a reader would write down.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import etree

from .apparatus import APPARATUS_TAGS, strip_apparatus, text_of


class CorpusParseError(ValueError):
    """A snapshot could not be parsed, with the file -- and where possible the section.

    An XMLSyntaxError from lxml names a line and column in a string nobody can find again.
    A build over 26 parts and 200 snapshots aborting on one of them has to say which one, or
    the only way to locate it is to bisect the cache by hand.
    """


def _parser() -> etree.XMLParser:
    """The parse settings, stated rather than inherited.

    ``resolve_entities=False`` is the load-bearing one. eCFR XML uses only the predefined
    entities and numeric character references, both of which still work; what it stops is an
    external or internal entity definition being expanded, which is both the XXE read
    primitive and the billion-laughs amplifier. ``no_network`` and ``huge_tree`` happen to
    match libxml2's current defaults, and are set anyway: a security property that holds
    because of another project's default is a property this code does not have.

    Built per call. lxml parsers carry state and are not safe to share across threads, and
    the construction cost is nothing against parsing a megabyte of XML.
    """
    return etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)


#: Leading paragraph designator: (a), (a)(1), (b)(2)(i) ...
_LABEL = re.compile(r"^\s*\(([a-zA-Z0-9]{1,4})\)((?:\s*\([a-zA-Z0-9]{1,4}\))*)")
_WS = re.compile(r"\s+")
#: eCFR heads read "§ 630.1203 Leave entitlement." -- keep the title, drop the number.
_HEAD_NUM = re.compile(r"^\s*(?:&#167;|§)?\s*[\d.\-]+\s*")


@dataclass(frozen=True)
class Paragraph:
    anchor: str  # "a", "a-1", or "p3" when the paragraph carries no designator
    text: str


@dataclass(frozen=True)
class Section:
    identifier: str  # "630.1203"
    heading: str
    text: str
    paragraphs: list[Paragraph] = field(default_factory=list)
    subpart: str | None = None

    @property
    def citation(self) -> str:
        return f"{self.identifier}"


_ROMAN = re.compile(r"^[ivxlcdm]+$")
_ROMAN_VALUE = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def _roman_to_int(token: str) -> int | None:
    if not _ROMAN.match(token):
        return None
    total = 0
    for i, ch in enumerate(token):
        v = _ROMAN_VALUE[ch]
        nxt = _ROMAN_VALUE.get(token[i + 1]) if i + 1 < len(token) else None
        total += -v if nxt and nxt > v else v
    return total


def _alpha_to_int(token: str) -> int | None:
    """Spreadsheet-style ordinal: a=1 ... z=26, aa=27.

    The doubled form is not hypothetical: §890.201 runs (a) through (z) and on to (gg).
    """
    low = token.lower()
    if not low.isalpha():
        return None
    n = 0
    for ch in low:
        n = n * 26 + (ord(ch) - 96)
    return n


#: (designator, kind) per level, deepest last.
_Stack = tuple[tuple[str, str], ...]

#: What a level is numbered with, given what its parent was numbered with: (a) contains (1)
#: contains (i) contains (A), and below that the CFR repeats (1) and (i). Kind is the only
#: thing that separates the ninth letter from the first roman numeral, so this map is what
#: makes ``(i)`` decidable at all.
#:
#: Keyed on the parent rather than on absolute depth, because a section is free to start
#: partway down: §330.602 and §551.104 are definitions whose lists open at (1) with no
#: lettered paragraph above them. Reading kind off the depth put a roman numeral two levels
#: under those, minting §330.602(1)(1)(i) for a paragraph the regulation calls (1)(i).
_BELOW = {"alpha": "digit", "digit": "roman", "roman": "upper", "upper": "digit"}
_ROOT_KIND = "alpha"

#: The designator each kind of level opens with.
_FIRST = {"alpha": "a", "digit": "1", "roman": "i", "upper": "A"}


def _kind_below(stack: _Stack) -> str:
    return _BELOW[stack[-1][1]] if stack else _ROOT_KIND


def _forms(token: str) -> dict[str, int]:
    """Every kind of level this designator could belong to, with its ordinal in each.

    ``(i)`` is the ninth letter *and* the first roman numeral, and the CFR uses both -- 5 CFR
    890.301 has (h)(1)(i) and a top-level (i) in the same section. The token cannot decide;
    all it can do is say which readings are open.
    """
    if token.isdigit():
        return {"digit": int(token)}
    if not token.isalpha():
        return {}
    if token.isupper():
        n = _alpha_to_int(token)
        return {"upper": n} if n is not None else {}
    if not token.islower():
        return {}
    out: dict[str, int] = {}
    roman = _roman_to_int(token)
    if roman is not None:
        out["roman"] = roman
    if len(set(token)) == 1:  # (a) .. (z), then (aa) .. (zz); never (ab)
        alpha = _alpha_to_int(token)
        if alpha is not None:
            out["alpha"] = alpha
    return out


def _is_successor(token: str, previous: str, kind: str) -> bool:
    """Is ``token`` the next designator after ``previous``, read as a ``kind`` level?

    The kind has to be supplied. Asking whether ``(ii)`` follows ``(i)`` without it is the
    question that produced the bug this signature exists to close: the answer is yes as roman
    numerals and no as letters, and a predicate that guesses will guess wrong somewhere.
    """
    a, b = _forms(token).get(kind), _forms(previous).get(kind)
    return a is not None and b is not None and a == b + 1


# How much each way of placing a designator costs. These are not thresholds anyone tuned to a
# score; they are an ordering, and only the order matters. Continuing an open level and
# opening the next one with its own first designator are what the CFR does by default and cost
# nothing. Everything below them is an irregularity, and the ranking says which irregularity
# is likelier: a chapeau that ran its first child into its own sentence (very common in eCFR)
# beats an entire level going unwritten, which beats a designator missing from a level, which
# beats a level numbered outside the CFR's cycle, which beats the same level restarting under
# one parent. `_resolve` sums them over a whole section, so a locally attractive reading that
# wrecks the next ten paragraphs loses to the one that does not.
_CONTINUE = 0
_OPEN = 0
_OPEN_INLINE = 2  # "(f) Open season. (1) ..." then a standalone "(2)"
_IMPLIED = 2      # ... and then a standalone "(i)", with the whole of (1) still inline
                  # -- charged per level buried in the parent's sentence
_SKIP = 5         # a gap in a level's numbering
_OFF_CYCLE = 6    # a level numbered with something other than what its parent calls for
_OFF_CYCLE_INLINE = 8
_RESTART = 9      # a level starting over under the same parent: an address collision
_DISPLACE = 12    # a designator no reading admits
_ORPHAN = 2       # a level opening on the far side of an undesignated paragraph


def _placements(stack: _Stack, token: str,
                orphaned: bool = False) -> list[tuple[int, _Stack]]:
    """Every level ``token`` could sit at, with what that reading costs.

    Replaces a greedy push. The greedy version scanned the stack for the first level the token
    continued and took it, which is why a top-level ``(i)`` after ``(h)(1)`` was unrecoverable:
    ``i`` continues ``h`` two levels up, and the reading that opens a roman level under (1) was
    never generated, let alone compared. Both are produced here and `_resolve` picks between
    them on what follows.

    ``orphaned`` says an undesignated paragraph stands between this stack and the token. A
    list that carries on across one is ordinary -- flush text closes a list and the list
    continues -- but a list that *starts* across one usually belongs to the undesignated
    paragraph rather than to the designator above it, so only opening pays for it.
    """
    forms = _forms(token)
    if not forms:
        depth = max(len(stack) - 1, 0)
        return [(_DISPLACE, stack[:depth] + ((token, _kind_below(stack[:depth])),))]

    out: list[tuple[int, _Stack]] = []
    for depth, (previous, kind) in enumerate(stack):
        here, there = forms.get(kind), _forms(previous).get(kind)
        if here is None or there is None:
            continue
        if here == there + 1:
            cost = _CONTINUE
        elif here > there + 1:
            cost = _SKIP
        elif here == 1:
            cost = _RESTART
        else:
            continue
        out.append((cost, stack[:depth] + ((token, kind),)))

    wanted = _kind_below(stack)
    orphan = _ORPHAN if orphaned else 0
    for kind, ordinal in forms.items():
        if kind == wanted:
            cost = _OPEN if ordinal == 1 else _OPEN_INLINE
        else:
            cost = _OFF_CYCLE if ordinal == 1 else _OFF_CYCLE_INLINE
        out.append((cost + orphan, stack + ((token, kind),)))

    # Levels written entirely inside their parent: "(e) Decreasing enrollment type. (1)
    # Subject to two exceptions ..." is followed by a standalone "(i)", and (e)(1) never gets
    # a <P> of its own. Filling it in is what makes 890.301#e-1-i the address a reader would
    # write. Two are allowed because §315.612(e) runs "(e) Proof of eligibility. (1)(i) Prior
    # to appointment ..." and then a standalone "(A)", burying both (1) and (i). Only from a
    # non-empty stack: with nothing above it a numbered top level -- which §550.1104 really
    # has -- would be pushed under an invented "(a)".
    filled = stack
    kind = wanted
    for implied in (1, 2):
        if not stack:
            break
        filled += ((_FIRST[kind], kind),)
        kind = _BELOW[kind]
        ordinal = forms.get(kind)
        if ordinal is None:
            continue
        cost = _IMPLIED * implied + (_OPEN if ordinal == 1 else _OPEN_INLINE)
        out.append((cost + orphan, filled + ((token, kind),)))
    return out


#: How many readings of a section's numbering are carried forward at once. The ambiguity is
#: local -- one or two designators deep -- and the widest section in the corpus, §890.301 with
#: 52 paragraphs, resolves at a width of 4. 16 is slack, and costs nothing measurable.
_BEAM = 16


def _resolve(items: list[tuple[list[str] | None, str]]) -> list[str]:
    """Anchors for a section's designator chains, chosen over the whole section at once.

    Paragraph by paragraph the numbering is ambiguous and no amount of care at one paragraph
    fixes it: after ``(h)(1)``, a ``(i)`` is equally the roman numeral opening (h)(1)(i) and
    the letter opening a new top-level (i), and 5 CFR 890.301 contains both readings, four
    paragraphs apart. What distinguishes them is what comes next -- ``(ii)`` continues the
    roman run, ``(1)`` cannot follow a roman numeral without a level being skipped -- so the
    decision is deferred and settled by the cheapest reading of the whole sequence.

    ``items`` is the section's whole body in order: a designator chain, or ``None`` and the
    positional anchor of an undesignated paragraph. The undesignated ones are not passengers.
    A definitions section is a run of headwords -- "*Employee* means a person who is
    employed--" -- each followed by its own (1), (2), (3), and reading those as children of
    the last designator seen produced §551.104(6)(1) and §630.201(b)(7)(1), addresses that
    look like citations and are not in the regulation. Each undesignated paragraph is
    therefore offered as an alternative root, and the search decides whether the list that
    follows belongs to it or to the designator above it.

    A beam rather than an exhaustive search: cost differences show up within a designator or
    two, and carrying every stack the section could have would be exponential for no gain.
    """
    # (cost, stack, index of the state it came from, anchor if a paragraph ends here)
    states: list[tuple[int, _Stack, int, str]] = [(0, (), -1, "")]
    beam = [0]
    answer = beam
    rooted = -1  # the state rooted at the most recent undesignated paragraph
    orphaned = False
    for chain, fallback in items:
        if chain is None:
            # Only the nearest undesignated paragraph is a candidate parent; an earlier one
            # is a headword whose own list has already been read or was never there.
            # Carries the cheapest reading so far as its history, and its cost, so that
            # re-rooting neither loses the anchors already assigned nor buys an advantage.
            prefix = min(beam, key=lambda i: (states[i][0], states[i][1]))
            states.append((states[prefix][0], ((fallback, _ROOT_KIND),), prefix, ""))
            beam = [i for i in beam if i != rooted] + [len(states) - 1]
            rooted, orphaned = len(states) - 1, True
            continue
        for position, token in enumerate(chain):
            ends = position == len(chain) - 1
            best: dict[_Stack, int] = {}
            for index in beam:
                cost, stack, _, _ = states[index]
                for extra, moved in _placements(stack, token,
                                                orphaned and index != rooted):
                    total = cost + extra
                    seen = best.get(moved)
                    if seen is not None and states[seen][0] <= total:
                        continue
                    anchor = "-".join(t for t, _ in moved) if ends else ""
                    states.append((total, moved, index, anchor))
                    best[moved] = len(states) - 1
            # Ties are broken on the stack itself so the same input always gives the same
            # anchors, whatever order the dict happened to be built in.
            beam = sorted(best.values(), key=lambda i: (states[i][0], states[i][1]))[:_BEAM]
            rooted, orphaned = -1, False
        answer = beam

    out: list[str] = []
    index = min(answer, key=lambda i: (states[i][0], states[i][1]))
    while index > 0:
        _, _, previous, anchor = states[index]
        if anchor:
            out.append(anchor)
        index = previous
    out.reverse()
    return out


#: Body elements that carry regulatory prose. ``P`` is the ordinary paragraph; the ``FP``
#: family is a *flush paragraph* -- unindented continuation text, used for the closing
#: sentence of a list and for the notes under a table. Reading only ``P`` silently dropped
#: 18,705 words, 4.5% of the corpus, concentrated in the Federal Wage System parts the
#: applicability story is built on: 88% of §532.313 and 46% of §531.214 were simply absent.
#:
#: That loss was invisible to the failure budget by construction. Its ``ingestion`` row asks
#: whether a gold chunk is in the store, and gold chunks are minted by this same function --
#: so text this parser never emitted could never be missed. A row that can only read zero is
#: not measuring anything, which is why the coverage assertion in tests/invariants exists.
_PROSE_TAGS = frozenset({"P", "FP", "FP-1", "FP-2", "FP1-2", "FP-DASH", "PSPACE"})
_TABLE_TAGS = frozenset({"TABLE", "GPOTABLE"})


def _table_text(node: etree._Element) -> str:
    """Flatten a table to one line per row, cells separated by ' | '.

    Serialised rather than skipped or split. A regulatory table is a single semantic unit --
    a wage schedule, a step progression -- and splitting it per cell destroys the row
    relationship that makes it answerable at all.
    """
    rows: list[str] = []
    for tr in node.iter("TR"):
        cells = [_WS.sub(" ", "".join(td.itertext())).strip()
                 for td in tr.iter("TD", "TH", "ENT")]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    if not rows:
        rows = [_WS.sub(" ", "".join(node.itertext())).strip()]
    return "\n".join(r for r in rows if r)


def _body_elements(section: etree._Element):
    """Prose and table elements in document order, without descending into them.

    A plain ``iter("P")`` would miss flush paragraphs and tables; iterating several tags
    naively would double-count the paragraphs nested inside an ``EXTRACT``. Walking and
    pruning at each body element gives each piece of text exactly once.
    """
    stack = list(reversed(list(section)))
    while stack:
        el = stack.pop()
        if el.tag in _PROSE_TAGS or el.tag in _TABLE_TAGS:
            yield el
            continue
        if el.tag in ("HEAD", *APPARATUS_TAGS):
            continue
        stack.extend(reversed(list(el)))


def _paragraphs(node: etree._Element) -> list[Paragraph]:
    """Paragraphs with hierarchical, section-unique anchors.

    Anchors must be unique inside a section version or a citation does not identify
    anything. Before the designator stack was tracked, 13% of addresses in the corpus were
    ambiguous -- ``550.703#a`` matched four different paragraphs, because a section with
    several sub-lists restarts at ``(a)`` and ``(1)`` repeatedly. A collision suffix is kept
    as a backstop for markup the stack cannot resolve; it should stay unused, and a test
    asserts uniqueness over the real corpus.
    """
    # Read first, address second. A designator's level depends on designators that have not
    # been read yet (see ``_resolve``), so the whole section's numbering is collected before
    # any of it is turned into an address.
    body: list[tuple[str, str, list[str] | None]] = []
    items: list[tuple[list[str] | None, str]] = []
    tables = 0
    for i, p in enumerate(_body_elements(node), start=1):
        if p.tag in _TABLE_TAGS:
            text = _table_text(strip_apparatus(p))
            if not text:
                continue
            tables += 1
            body.append((f"t{tables}", text, None))
            continue
        text = _WS.sub(" ", "".join(strip_apparatus(p).itertext())).strip()
        if not text:
            continue
        m = _LABEL.match(text)
        if not m:
            # Flush text with no designator: an introductory or concluding paragraph.
            body.append((f"p{i}", text, None))
            items.append((None, f"p{i}"))
            continue
        chain = [m.group(1), *re.findall(r"\(([a-zA-Z0-9]{1,4})\)", m.group(2) or "")]
        body.append(("", text, chain))
        items.append((chain, ""))

    anchors = iter(_resolve(items))
    out: list[Paragraph] = []
    used: dict[str, int] = {}
    for fallback, text, chain in body:
        anchor = next(anchors) if chain is not None else fallback
        if anchor in used:
            used[anchor] += 1
            anchor = f"{anchor}.{used[anchor]}"
        else:
            used[anchor] = 1
        out.append(Paragraph(anchor=anchor, text=text))
    return out


def _subpart_of(node: etree._Element) -> str | None:
    for anc in node.iterancestors("DIV6"):
        if anc.get("TYPE") == "SUBPART":
            return anc.get("N")
    return None


def parse_sections(xml: bytes, *, source: str | None = None) -> list[Section]:
    """Every section in a part snapshot, apparatus already removed.

    ``source`` names the snapshot in any error raised -- pass the cache filename or the
    title/part/date. Ingestion walks 26 parts and 200 snapshots, so a failure that does not
    name its input is a failure nobody can reproduce.
    """
    where = source or f"<{len(xml)} bytes of XML>"
    if not xml.strip():
        raise CorpusParseError(f"{where}: empty snapshot, nothing to parse")
    try:
        root = etree.fromstring(xml, _parser())
    except etree.XMLSyntaxError as exc:
        raise CorpusParseError(f"{where}: not well-formed XML ({exc})") from exc
    if root is None:
        raise CorpusParseError(f"{where}: no document element")

    sections: list[Section] = []
    for div in root.iter("DIV8"):
        if div.get("TYPE") != "SECTION":
            continue
        ident = (div.get("N") or "").strip()
        if not ident:
            continue
        try:
            head_el = div.find("HEAD")
            heading = ""
            if head_el is not None:
                heading = _HEAD_NUM.sub("", _WS.sub(" ", "".join(head_el.itertext())).strip())
            sections.append(
                Section(
                    identifier=ident,
                    heading=heading.strip(" .§"),
                    text=text_of(div),
                    paragraphs=_paragraphs(div),
                    subpart=_subpart_of(div),
                )
            )
        except (etree.LxmlError, ValueError) as exc:
            raise CorpusParseError(f"{where}: section {ident} could not be read "
                                   f"({type(exc).__name__}: {exc})") from exc
    return sections


def section_index(xml: bytes, *, source: str | None = None) -> dict[str, Section]:
    return {s.identifier: s for s in parse_sections(xml, source=source)}
