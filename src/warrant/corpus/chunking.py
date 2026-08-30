"""What a *retrievable* unit is -- a policy layer over what parse.py extracts.

parse.py reads the regulation's own hierarchy and stops there. This module decides which of
those elements is worth ranking, what has to travel with one to make it rankable at all, and
what the generator should be handed once it is ranked. The split is deliberate: every
decision here is a pure function of a ``Section``, so a chunking change can be argued about
against a hand-built section rather than against a 10,000-row store.

Measured on the in-force corpus (9,961 chunks) under the previous policy, which was one
chunk per paragraph element and nothing else:

    mean 40.0 tokens - median 31 - p10 10 - p90 79 - max 1013
    under 10 tokens:   960  (9.6%)
    under 20 tokens: 2,939 (29.5%)
    under 30 tokens: 4,806 (48.2%)

Nearly half the corpus was too short to retrieve on its own words. ``300.102#b`` is the whole
of "(b) Result in selection from among the best qualified candidates;" -- ten tokens naming
neither the subject they qualify nor the section they sit in -- and ``300.401#p1`` is "For
purposes of this subpart:", a chapeau whose definitions were stored as if they hung off
nothing. Four policies answer that, each of which leaves ``Chunk.text`` verbatim, because
that is the text a citation points at:

    context   the subpart, the section heading, the chapeau, and every governing designator
    parents   the enclosing unit, so ranking can be precise and answering can still be whole
    merge     a unit under ``min_tokens`` retrieves as its parent and cites as itself
    split     a unit over ``max_tokens`` becomes several, cut at sentence ends, never a table

``ChunkPolicy.legacy()`` turns all four off and reproduces the previous behaviour exactly, so
the two are comparable on one store rather than on one recollection.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .parse import Paragraph, Section

if TYPE_CHECKING:  # pragma: no cover
    from ..index.store import Store


def token_count(text: str) -> int:
    """Whitespace tokens, which is what the distribution above is quoted in.

    Not the encoder's tokenizer. A policy that could only be evaluated by loading a 130 MB
    model would not be evaluable on a fresh clone, and the two counts differ by a roughly
    constant factor (~1.4x for bge-small on this corpus) that no threshold here is sensitive
    to. The one place the difference matters is ``max_tokens``, which is set with the
    factor in mind -- see ``ChunkPolicy``.
    """
    return len(text.split())


@dataclass(frozen=True)
class ChunkPolicy:
    """The four decisions, and the thresholds they turn on.

    ``max_tokens`` is 200 rather than "as large as fits". 200 whitespace tokens plus context
    is roughly 300 wordpiece tokens, inside the 512-token window of both bge-small-en-v1.5
    and ms-marco-MiniLM-L-6-v2; §353.203(c) at 331 tokens is not, and the tail of it was
    being silently truncated by the reranker that was supposed to be judging it. Only 45 of
    9,961 in-force chunks exceed 200, so this is a tail fix, not a re-segmentation.

    ``min_tokens`` is 15 because that is just above the p10 of 10: it catches the 1,899
    chunks (19.1%) whose own words cannot identify them, without pulling in the median.
    """

    min_tokens: int = 15
    max_tokens: int = 200
    #: Sentences of the previous piece repeated at the head of the next. One is enough to
    #: keep a pronoun or a "such employee" attached to its antecedent; more would duplicate
    #: enough text that two pieces of one paragraph compete for the same result slot.
    overlap_sentences: int = 1
    #: Ceiling on the governing text pulled into ``context``. Context is prepended to every
    #: ranking stage's input, so an unbounded one would push the paragraph itself out of the
    #: encoder window -- the exact failure ``max_tokens`` exists to prevent.
    max_context_tokens: int = 120
    context: bool = True
    parents: bool = True
    merge: bool = True
    split: bool = True

    @classmethod
    def legacy(cls) -> ChunkPolicy:
        """One chunk per paragraph element, no context, no parents. The previous behaviour.

        Kept so the change can be measured rather than asserted: rebuild twice, score twice,
        and the delta is over two stores built by the same code from the same snapshots.
        """
        return cls(context=False, parents=False, merge=False, split=False)


DEFAULT_POLICY = ChunkPolicy()


@dataclass(frozen=True)
class Unit:
    """One retrievable unit: a citation address, verbatim text, and what governs it.

    ``text`` is never touched. It is what a reader would look up at ``anchor``, and a
    citation to text that was rewritten on the way into the store is not a citation.
    Everything the policy adds lives in ``context``, which retrieval reads and citation
    does not.
    """

    anchor: str | None            # None only for a section that parsed to no paragraphs
    text: str                     # verbatim, always
    context: str = ""
    parent_id: str = ""
    kind: str = "prose"           # "prose" or "table"
    merged: bool = False          # retrieves as its parent, cites as itself
    split_from: str | None = None  # the anchor this piece was cut out of

    @property
    def tokens(self) -> int:
        return token_count(self.text)


#: parse.py's only surviving signal that a paragraph element was a ``TABLE``/``GPOTABLE`` is
#: the anchor it minted for it: ``t1``, ``t2``, with ``.2`` appended on collision. Reading it
#: back here rather than threading a tag through ``Paragraph`` keeps parse.py's dataclass the
#: neutral record it is; the cost is this coupling, which one test pins.
_TABLE_ANCHOR = re.compile(r"^t\d+(\.\d+)?$")
#: An undesignated paragraph -- parse.py's ``p{index}`` -- is flush text with no designator:
#: an introduction, a chapeau, or a concluding sentence.
_FLUSH_ANCHOR = re.compile(r"^p\d+(\.\d+)?$")
#: A subpart designator worth naming. DIV6/@N is usually "A".."Q", but 1 of the 26 parts
#: carries a generated id (``ECFRd94e3f811e1d5af``) instead, and "Subpart ECFRd94e3f81..." in
#: the text every ranking stage sees is noise with a token cost.
_SUBPART_ID = re.compile(r"^[A-Za-z0-9]{1,4}$")


def is_table(anchor: str | None) -> bool:
    return bool(anchor and _TABLE_ANCHOR.match(anchor))


def _is_flush(anchor: str | None) -> bool:
    return bool(anchor and _FLUSH_ANCHOR.match(anchor))


def _designator_path(anchor: str) -> list[str]:
    """The anchors that govern ``anchor``, outermost first: b-1-ii -> ['b', 'b-1'].

    The collision suffix parse.py appends (``a-1.2``) is not part of the hierarchy and is
    stripped before splitting; the suffixed anchor is a different paragraph at the same
    address, not a child of one.
    """
    base = anchor.split(".", 1)[0]
    parts = base.split("-")
    return ["-".join(parts[:i]) for i in range(1, len(parts))]


# -- sentence boundaries ----------------------------------------------------------

#: A break candidate: sentence punctuation, whitespace, then something that can open a
#: sentence. The lookahead is what keeps "e.g. the agency" and "31.5 percent" intact.
_BREAK = re.compile(r"(?<=[.?!])\s+(?=[\"'(\[A-Z§])")
#: Semicolons are the CFR's real list separator, and a 300-token paragraph is usually one
#: sentence with a dozen of them. Used only to break a sentence that is itself oversized.
_SEMI_BREAK = re.compile(r";\s+")
_TRAILING_WORD = re.compile(r"([A-Za-z.]+)\.$")
#: Words that end in a period without ending a sentence. Single letters (initials, "Pub. L.")
#: are rejected by length rather than listed.
_ABBREVIATIONS = frozenset({
    "usc", "cfr", "fr", "no", "nos", "sec", "secs", "subsec", "pub", "app", "cf", "eg",
    "ie", "etc", "vs", "st", "mr", "mrs", "ms", "dr", "jr", "sr", "inc", "co", "dept",
    "fig", "para", "paras", "pt", "subpt", "ch", "art", "min", "max", "approx", "est",
})


def _ends_an_abbreviation(text: str, end: int) -> bool:
    m = _TRAILING_WORD.search(text[:end])
    if m is None:
        return False
    word = m.group(1).rstrip(".")
    return len(word) == 1 or word.replace(".", "").lower() in _ABBREVIATIONS


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Sentence boundaries as ``(start, end)`` offsets into ``text``.

    Offsets rather than strings, so every piece cut from them is a literal slice of the
    original. Rejoining split tokens with a space would *usually* reproduce the input --
    parse.py has already collapsed whitespace -- and "usually verbatim" is not a property a
    citation can rest on.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _BREAK.finditer(text):
        if _ends_an_abbreviation(text, m.start()):
            continue
        spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans or [(0, len(text))]


def _subdivide(text: str, spans: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    """Break any single sentence that is on its own over ``limit``, at its semicolons."""
    out: list[tuple[int, int]] = []
    for a, b in spans:
        if token_count(text[a:b]) <= limit:
            out.append((a, b))
            continue
        cut = a
        for m in _SEMI_BREAK.finditer(text, a, b):
            out.append((cut, m.start() + 1))   # keep the ';' with the clause it closes
            cut = m.end()
        out.append((cut, b))
    return [(a, b) for a, b in out if text[a:b].strip()]


def split_spans(text: str, *, max_tokens: int, overlap_sentences: int) -> list[tuple[int, int]]:
    """Pack sentences into pieces of at most ``max_tokens``, overlapping by whole sentences.

    Greedy rather than balanced: a balanced split would move the boundary of every piece
    when one sentence is amended, and the anchors are citation addresses that should stay
    put across snapshots wherever the text did not change.
    """
    spans = _subdivide(text, sentence_spans(text), max_tokens)
    if len(spans) == 1:
        return [(spans[0][0], spans[0][1])]
    pieces: list[tuple[int, int]] = []
    i = 0
    while i < len(spans):
        j, total = i, 0
        while j < len(spans):
            n = token_count(text[spans[j][0]:spans[j][1]])
            if j > i and total + n > max_tokens:
                break
            total += n
            j += 1
        pieces.append((spans[i][0], spans[j - 1][1]))
        if j >= len(spans):
            break
        # Step back by the overlap, but always forward by at least one sentence: an overlap
        # as large as the piece would otherwise emit the same span until the heat death.
        i = max(i + 1, j - overlap_sentences)
    return pieces


# -- context ----------------------------------------------------------------------


def _first_sentence(text: str, *, limit: int) -> str:
    """The opening sentence, capped. Ancestors are quoted for what they govern, not read."""
    a, b = sentence_spans(text)[0]
    head = text[a:b].strip()
    words = head.split()
    return head if len(words) <= limit else " ".join(words[:limit]) + " ..."


def _section_lines(section: Section) -> list[str]:
    lines = []
    if section.subpart and _SUBPART_ID.match(section.subpart):
        lines.append(f"Subpart {section.subpart}")
    heading = (section.heading or "").strip()
    lines.append(f"§ {section.identifier} {heading}".rstrip())
    return lines


def _chapeaux(paragraphs: Sequence[Paragraph]) -> dict[str, str]:
    """anchor -> the anchor of the flush chapeau it hangs off, if any.

    A chapeau is recognised by its colon. Flush text appears at both ends of a list -- "For
    purposes of this subpart:" opens one, "The agency shall document the decision." closes
    one -- and parse.py gives both the same ``p{index}`` anchor, so position alone cannot
    tell them apart. 159 of the 1,406 undesignated in-force paragraphs end in a colon;
    treating the other 1,247 as chapeaux would attach concluding sentences to paragraphs
    they do not govern.
    """
    out: dict[str, str] = {}
    current: str | None = None
    for p in paragraphs:
        if current:
            out[p.anchor] = current
        if _is_flush(p.anchor):
            current = p.anchor if p.text.rstrip().endswith(":") else None
    return out


def _descendants(anchor: str, paragraphs: Sequence[Paragraph]) -> list[Paragraph]:
    """Every paragraph governed by ``anchor``, in document order."""
    prefix = f"{anchor}-"
    return [p for p in paragraphs if p.anchor.split(".", 1)[0].startswith(prefix)]


# -- the policy -------------------------------------------------------------------


def units(section: Section, policy: ChunkPolicy = DEFAULT_POLICY) -> list[Unit]:
    """The retrievable units of one section version, in document order.

    A section that parsed to no paragraphs still yields exactly one unit carrying the whole
    section text. Dropping it would make the section unretrievable, which is the silent
    corpus hole the ingestion row of the failure budget exists to expose -- and cannot see,
    because it reads the same parser.
    """
    paragraphs = list(section.paragraphs or [])
    if not paragraphs:
        context = "\n".join(_section_lines(section)) if policy.context else ""
        return [Unit(anchor=None, text=section.text, context=context)]

    by_anchor = {p.anchor: p for p in paragraphs}
    chapeau = _chapeaux(paragraphs) if policy.context else {}
    out: list[Unit] = []

    for p in paragraphs:
        table = is_table(p.anchor)
        ancestors = [a for a in _designator_path(p.anchor) if a in by_anchor]
        parent = ancestors[-1] if ancestors else None
        # No parent paragraph means the enclosing unit is the section itself, addressed
        # bare. Small-to-big has to bottom out somewhere, and the section is the unit
        # parse.py already declares the retrieval unit.
        enclosing = f"{section.identifier}#{parent}" if parent else section.identifier

        # A table is one semantic unit. It is never merged into the prose around it -- the
        # row relationship is the meaning, and prose glued to it reads as another row -- and
        # never split, for the same reason parse.py refuses to split it per cell.
        merged = bool(policy.merge and policy.context and not table
                      and token_count(p.text) < policy.min_tokens)

        lines: list[str] = []
        if policy.context:
            lines = _section_lines(section)
            # One budget across everything that governs this paragraph, spent outermost
            # first. Per-line caps would multiply by depth, and a four-deep designator would
            # arrive at the encoder with its own text pushed out of the window.
            budget = policy.max_context_tokens
            head = chapeau.get(p.anchor)
            governing = ([head] if head else []) + ancestors
            quoted = {p.anchor, *governing}
            for a in governing:
                line = _first_sentence(by_anchor[a].text, limit=budget)
                lines.append(line)
                budget -= token_count(line)
                if budget <= 0:
                    break
            if merged and budget > 0:
                lines.extend(_merged_body(p, parent, paragraphs, budget, quoted))

        pieces = [(0, len(p.text))]
        if policy.split and not table and token_count(p.text) > policy.max_tokens:
            pieces = split_spans(p.text, max_tokens=policy.max_tokens,
                                 overlap_sentences=policy.overlap_sentences)

        for i, (a, b) in enumerate(pieces, start=1):
            piece_lines = list(lines)
            anchor = p.anchor
            if len(pieces) > 1:
                # Zero-padded so the addresses of a nine-way split still sort in reading
                # order, which is how they are read back and how a reviewer scans them.
                anchor = f"{p.anchor}.s{i:02d}"
                if i > 1 and policy.context:
                    # A continuation no longer carries its own designator or subject:
                    # "(c) Nature of Reserve service." is in piece one only.
                    piece_lines.append(_first_sentence(p.text,
                                                       limit=policy.max_context_tokens))
            out.append(Unit(anchor=anchor, text=p.text[a:b],
                            parent_id=enclosing if policy.parents else "",
                            context="\n".join(piece_lines),
                            kind="table" if table else "prose", merged=merged,
                            split_from=p.anchor if len(pieces) > 1 else None))
    return out


def _merged_body(unit: Paragraph, parent: str | None, paragraphs: Sequence[Paragraph],
                 budget: int, quoted: set[str]) -> list[str]:
    """The rest of the enclosing unit, for a paragraph too short to retrieve alone.

    This is the section/paragraph split parse.py already declares -- section the retrieval
    unit, paragraph the citation unit -- applied where it was not. The paragraph keeps its
    own anchor and its own verbatim text; what changes is only the string retrieval ranks.

    Taken in document order and stopped at the budget, so several short siblings share one
    stable retrieval unit instead of each computing a different window around itself.
    ``quoted`` is what the ancestor lines already carry, so the chapeau is not repeated.
    Tables are skipped: a wage schedule flattened into a definition's context is a wall of
    pipes that outweighs the definition it was meant to explain.
    """
    body = _descendants(parent, paragraphs) if parent else paragraphs
    lines: list[str] = []
    for sib in body:
        if sib.anchor in quoted or is_table(sib.anchor):
            continue
        n = token_count(sib.text)
        if n > budget:
            break
        lines.append(sib.text)
        budget -= n
    return lines


# -- small-to-big -----------------------------------------------------------------


@dataclass(frozen=True)
class ParentUnit:
    """One enclosing unit to hand a generator, and the retrieved chunks that asked for it."""

    parent_id: str
    text: str
    requested_by: tuple[str, ...]


def parent_texts(store: Store, chunk_ids: Sequence[str], *, valid_date: str,
                 system_time: str | None = None) -> list[ParentUnit]:
    """The enclosing units behind a ranked list, deduplicated, in the order first requested.

    Deduplication is the whole point of returning a list of parents rather than a parent per
    hit. Ranking a paragraph list works precisely because the siblings are separate rows, and
    every one of those siblings names the same parent -- eight results from one subsection
    would otherwise hand the generator the same section eight times and spend the context
    window proving it.

    Dated, because the parent has to be the version that was in force alongside the child. A
    parent fetched by ``chunk_id`` alone would happily return the 2024 text of a paragraph
    retrieved as of 2018, which is the single failure this store exists to prevent.
    """
    from ..index.store import now

    if not chunk_ids:
        return []
    sys_t = system_time or now()
    valid = ("valid_from <= :v AND (valid_to IS NULL OR valid_to > :v) "
             "AND system_from <= :s AND (system_to IS NULL OR system_to > :s)")
    marks = ",".join(f":c{i}" for i in range(len(chunk_ids)))
    params: dict[str, object] = {"v": valid_date, "s": sys_t}
    params.update({f"c{i}": c for i, c in enumerate(chunk_ids)})
    rows = store.db.execute(
        f"SELECT chunk_id, parent_id FROM chunk WHERE chunk_id IN ({marks}) AND {valid}",
        params).fetchall()

    parents: dict[str, list[str]] = {}
    found = {r["chunk_id"]: r["parent_id"] for r in rows}
    for cid in chunk_ids:                      # requested order, not row order
        parent = found.get(cid)
        if parent:
            parents.setdefault(parent, []).append(cid)

    out: list[ParentUnit] = []
    for parent_id, asked in parents.items():
        if "#" in parent_id:
            got = store.db.execute(
                f"SELECT text FROM chunk WHERE chunk_id = :p AND {valid} ORDER BY id",
                {**params, "p": parent_id}).fetchall()
        else:
            # A bare parent id is a section: the enclosing unit is every paragraph of it,
            # reassembled in document order. ``id`` is insertion order, which is document
            # order, because ingestion writes a section's paragraphs in one executemany.
            got = store.db.execute(
                f"SELECT text FROM chunk WHERE section_id = :p AND {valid} ORDER BY id",
                {**params, "p": parent_id}).fetchall()
        text = "\n".join(r["text"] for r in got)
        if text:
            out.append(ParentUnit(parent_id, text, tuple(asked)))
    return out


# -- reporting --------------------------------------------------------------------

#: The thresholds the problem was stated in. Reported every time, so a run that improved the
#: mean while leaving the short tail alone cannot read as a win.
THRESHOLDS = (10, 20, 30)


@dataclass(frozen=True)
class Distribution:
    """Chunk sizes, in the shape a chunking change has to be argued in.

    Percentiles are nearest-rank, not interpolated: every number printed is the length of an
    actual chunk somebody can go and read, and "p90 79.5 tokens" describes nothing.
    """

    n: int
    mean: float
    median: int
    p10: int
    p90: int
    maximum: int
    under: dict[int, int]

    @classmethod
    def of(cls, counts: Iterable[int],
           thresholds: Sequence[int] = THRESHOLDS) -> Distribution:
        values = sorted(counts)
        if not values:
            return cls(0, 0.0, 0, 0, 0, 0, {t: 0 for t in thresholds})

        def rank(q: float) -> int:
            return values[max(0, math.ceil(q * len(values)) - 1)]

        return cls(n=len(values), mean=sum(values) / len(values), median=rank(0.5),
                   p10=rank(0.1), p90=rank(0.9), maximum=values[-1],
                   under={t: sum(1 for v in values if v < t) for t in thresholds})

    def share_under(self, threshold: int) -> float:
        return self.under.get(threshold, 0) / self.n if self.n else 0.0

    def __str__(self) -> str:
        head = (f"{self.n:,} chunks - mean {self.mean:.1f} tokens - median {self.median} - "
                f"p10 {self.p10} - p90 {self.p90} - max {self.maximum}")
        tail = "; ".join(f"under {t}: {self.under[t]:,} ({self.share_under(t):.1%})"
                         for t in sorted(self.under))
        return f"{head}\n{tail}"


def distribution(texts: Iterable[str],
                 thresholds: Sequence[int] = THRESHOLDS) -> Distribution:
    return Distribution.of((token_count(t) for t in texts), thresholds)


def store_distribution(store: Store, *, unit: str = "citation",
                       thresholds: Sequence[int] = THRESHOLDS,
                       valid_date: str | None = None) -> Distribution:
    """The size distribution of what is in force and believed, straight out of the store.

    Two populations, and reporting only one of them misreads the change entirely:

      ``citation``   ``Chunk.text``, what a citation points at. Contextual augmentation
                     leaves this almost untouched *by design* -- only splitting moves it --
                     so a flat citation distribution is the verbatim guarantee holding, not
                     a policy that did nothing.
      ``retrieval``  ``Chunk.retrieval_text``, what every ranking stage actually scores.
                     This is where merging and context show up, and it is the only one of
                     the two that the 9.6%-under-ten-tokens problem was ever about.

    ``valid_date`` defaults to "currently in force" -- open valid interval -- because that is
    the 9,961-chunk population every number in this module's docstring was measured on.
    Superseded versions are real chunks, but including them would mix several vintages of the
    same paragraph into one distribution and move it for reasons that are not policy.
    """
    if unit not in ("citation", "retrieval"):
        raise ValueError(f"unit must be 'citation' or 'retrieval', not {unit!r}")
    where = ("valid_to IS NULL AND system_to IS NULL" if valid_date is None else
             "valid_from <= :v AND (valid_to IS NULL OR valid_to > :v) AND system_to IS NULL")
    rows = store.db.execute(f"SELECT context, text FROM chunk WHERE {where}",
                            {} if valid_date is None else {"v": valid_date}).fetchall()
    if unit == "citation":
        return distribution((r["text"] for r in rows), thresholds)
    return distribution(
        (f"{r['context']}\n{r['text']}" if r["context"] else r["text"] for r in rows),
        thresholds)


def report(after: Distribution, before: Distribution | None = None) -> str:
    """Before and after in one block, ready for the CLI to print.

    A chunking change with no distribution printed is a change nobody can review: the whole
    argument for merging and splitting is a claim about the shape of the corpus, and the
    shape is cheap to measure and impossible to argue with.
    """
    if before is None:
        return str(after)
    lines = [f"before  {str(before).splitlines()[0]}",
             f"after   {str(after).splitlines()[0]}"]
    for t in sorted(after.under):
        lines.append(f"  under {t:>3} tokens: "
                     f"{before.under[t]:>6,} ({before.share_under(t):>5.1%})  ->  "
                     f"{after.under[t]:>6,} ({after.share_under(t):>5.1%})")
    return "\n".join(lines)
