"""The conditions a cited paragraph is bounded by, and whether the answer said them.

The failure this exists for is not the invented fact. Hallucination on the held-out human
items is 1.5% and citation precision 98.5%; what those numbers cannot see is the **true
sentence with the exception dropped**. "Restored annual leave must be scheduled for use
within 2 years" is a faithful quotation of §630.306(a) and a wrong answer, because (a) opens
"Except as provided in paragraph (b) of this section". Span alignment passes it. Entailment
passes it -- the premise really does entail the claim. Citation precision passes it. The
claim is simply not the whole rule.

Measured over the 9,961 in-force chunks, **25.7% carry at least one such bound**:

    chapeau                  979   9.8%      a lead-in governing an enumerated list
    except                   504   5.1%
    prohibition              484   4.9%      may / shall / must not
    subject_to               389   3.9%
    unless                   279   2.8%
    only_if                  117   1.2%
    other_than               113   1.1%
    proviso                   69   0.7%
    bound                     69   0.7%      a stated ceiling, not a condition
    notwithstanding           56   0.6%

So a qualifier is returned as an object with a kind, a character span and the addresses it
points at -- never as a boolean. "This chunk is conditional" is not actionable; "this chunk
is conditional on §630.310(d), which is not in the evidence set" is.

**Character spans are correct here and only here.** Citations in this repository are evidence
ids throughout, because the generator cannot count characters (ARCHITECTURE.md section 5).
Nothing in this module asks a model for an offset: the spans are produced by a regex over one
chunk's own text, which is arithmetic, and they are what lets the unstated-condition check
compare the answer against *the condition clause* rather than against the whole paragraph.

Two kinds carry their own warning. ``bound`` -- "may not exceed 30 days" -- is split out from
``prohibition`` because it is a stated number rather than an unstated condition; ``chapeau``
is structural rather than lexical, and is the case an all-regex detector would otherwise miss
entirely.

Precision is 90.6% (95% CI 85.6-94.6) on 104 hand-labelled instances, weighted by how often
each kind occurs. ``subject_to`` alone is 53% and is shipped untuned, because what separates
"subject to the provisions of paragraph (d)(iii)" from "employees subject to this subpart" is
what the phrase attaches to, and no lexical guard found that. Recall is the larger hole: the
bare ``if`` / ``when`` antecedent appears in 13.0% of chunks, is not in this trigger list, and
a 20-chunk probe of what the detector calls unconditioned found a condition in 11 of them.
All of it is in docs/results/eval-006-unstated-conditions.md, instance by instance.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .xref import anchor_of, chunk_id_of, find_references, section_of

#: Words too common to count as acknowledgement. Deliberately the same list ``align`` uses,
#: minus the modals and negations -- "not", "may", "shall", "such", "other", "under" are the
#: whole content of a condition and dropping them is how a check for conditions goes blind.
_STOP = frozenset("""
a an the and or of to in for on at by is are was were be been being as that this these those
it its with which any all from than then when
""".split())
_WORD = re.compile(r"[a-z0-9]+")

#: A qualifier clause runs from its trigger to the end of the clause it governs. Ending it at
#: a sentence boundary only would swallow the rule the exception qualifies, which is exactly
#: the text the acknowledgement check must not be allowed to match against.
_CLAUSE_END = re.compile(r";|(?<=[.])\s+(?=[(A-Z])|$")
#: A comma ends a *leading* qualifier -- "Except as provided in paragraph (b) of this section,
#: an employee ..." -- but not the commas inside its own citation run, "(b), (c), and (d)".
#: It does not end a trailing one: "only if OPM determines that the agency has, for a period
#: of no less than 90 days, ..." cut the condition in half at the first comma and left the
#: acknowledgement check scoring against four words.
_INNER_COMMA = re.compile(r",\s*(?:\(|and\b|or\b|\d{3}\.)")
_SENTENCE_START = re.compile(r"(?:^|[.;:]\s+|^\s*\([A-Za-z0-9]{1,4}\)\s*)$")

_TRIGGERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # "excepted service" is a category of appointment, not an exception. The word boundary is
    # enough: 126 chunks contain "excepted" and the trigger fires on exactly the 504 that
    # contain "except".
    ("except", re.compile(r"\bexcept\b", re.IGNORECASE)),
    ("unless", re.compile(r"\bunless\b", re.IGNORECASE)),
    ("notwithstanding", re.compile(r"\bnotwithstanding\b", re.IGNORECASE)),
    ("subject_to", re.compile(r"\bsubject to\b", re.IGNORECASE)),
    # Bare "provided" is a past participle 8 times in 10 in this corpus -- "the documentation
    # provided", "benefits provided under". A proviso is followed by its own subject.
    ("proviso", re.compile(
        r"\bprovided\s+(?:that\b|(?:the|such|a|an|he|she|it|they|his|her)\s+\w+\s+\w)",
        re.IGNORECASE)),
    ("only_if", re.compile(r"\bonly\s+(?:if|when|where|after|upon|to the extent|in)\b",
                           re.IGNORECASE)),
    ("prohibition", re.compile(r"\b(?:may|shall|must|will|can)\s+not\b|\bmay\s+no\s+longer\b",
                               re.IGNORECASE)),
    ("other_than", re.compile(r"\bother than\b", re.IGNORECASE)),
)

#: What turns a prohibition into a stated numeric bound. Measured: 57 of the 391 "may not"
#: chunks are "may not exceed", and treating them as unstated conditions put a qualifier on
#: every answer that already quoted the number. The "cause ... to exceed" arm came from the
#: labelled sample: §591.238(b) pays "so much of the post differential as will not cause the
#: combined total to exceed 25 percent", which is a ceiling written as a prohibition.
_BOUND = re.compile(
    r"^(?:may|shall|must|will|can)\s+not\s+"
    r"(?:be\s+)?(?:exceeds?\b|be\s+(?:less|more|greater|earlier|later|higher|lower)\b|"
    r"cause\b[^.;]{0,80}?\bto\s+exceed\b)",
    re.IGNORECASE)

#: "at a time other than the end of the contract year" and "in other than the full fraction"
#: are comparatives, not exclusions from the rule. The exclusion reading needs a noun the
#: rule applies *to* in front of it.
_OTHER_THAN_MANNER = re.compile(
    r"(?:\b(?:a|an|the)\s+(?:time|date|manner|way|form|fraction|basis|method|means|"
    r"place|purpose|amount|rate|order)\s*|\bin\s+)$",
    re.IGNORECASE)

#: A lead-in paragraph governing an enumerated list. eCFR renders the em dash of "is provided
#: without loss of--" as an em dash or a hyphen depending on the snapshot, so both are
#: accepted; a colon is the commoner form and accounts for 8.0% of in-force chunks on its own.
#: The colon is not sufficient by itself -- 300.201(a) ends in one and enumerates inline, so
#: nothing hangs off it in the store and there is no chapeau relationship to report.
_CHAPEAU_END = re.compile(r"[:—–]\s*$|\-\-\s*$")


@dataclass(frozen=True)
class Qualifier:
    """One condition bounding the chunk it was found in.

    ``span`` indexes the chunk's own text. ``refers_to`` holds the chunk ids named *inside*
    the qualifier clause -- for "Except as provided in paragraph (b) of this section" that is
    ``('630.306#b',)``, which is what makes a dropped exception traceable to the paragraph
    that would have stated it, rather than merely flagged.
    """

    kind: str                       # except | unless | subject_to | notwithstanding |
                                    # proviso | only_if | prohibition | bound | other_than |
                                    # chapeau
    span: tuple[int, int]
    text: str
    refers_to: tuple[str, ...] = ()

    @property
    def conditional(self) -> bool:
        """Does this bound the rule on something outside the sentence that states it?

        ``bound`` is excluded: a number the answer can simply repeat is not an unstated
        condition. ``chapeau`` is included -- an enumerated item read without its lead-in is
        the same failure wearing different punctuation.
        """
        return self.kind != "bound"


def _clause(text: str, start: int) -> int:
    """Where the clause beginning at ``start`` ends."""
    end = len(text)
    m = _CLAUSE_END.search(text, start)
    if m and m.start() > start:
        end = m.start()
    if not _SENTENCE_START.search(text[:start]):
        return end
    depth = 0
    for i in range(start, end):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and not depth and not _INNER_COMMA.match(text, i):
            return i
    return end


def qualifiers(text: str, *, chunk_id: str = "",
               enumerates: bool = False) -> list[Qualifier]:
    """Every condition bounding ``text``, in document order, one per trigger.

    ``chunk_id`` lets a reference inside a qualifier resolve "paragraph (b) of this section"
    against the citing section; without it the qualifier is still found and ``refers_to``
    comes back empty rather than guessed. ``enumerates`` says the store holds sub-paragraphs
    hanging off this chunk, which is what separates a real chapeau from a paragraph that
    happens to end in a colon.
    """
    section = section_of(chunk_id) if chunk_id else ""
    anchor = anchor_of(chunk_id) if chunk_id else ""
    taken: list[tuple[int, int]] = []
    found: list[Qualifier] = []
    for trigger, pattern in _TRIGGERS:
        for m in pattern.finditer(text):
            start, end = m.start(), _clause(text, m.start())
            # Two triggers at two positions are two conditions, even when one clause runs
            # through the other. Suppressing on clause overlap instead lost the 480-hour cap
            # in 630.401(c) to a "subject to" that opened earlier in the same sentence.
            if any(start < b and a < m.end() for a, b in taken):
                continue
            clause = text[start:end].strip()
            if not clause:
                continue
            kind = trigger
            if kind == "prohibition" and _BOUND.match(clause):
                kind = "bound"
            elif kind == "other_than" and _OTHER_THAN_MANNER.search(text[:start]):
                continue
            taken.append((start, m.end()))
            refs = find_references(clause, section_id=section, anchor=anchor)
            found.append(Qualifier(
                kind=kind, span=(start, end), text=clause,
                refers_to=tuple(dict.fromkeys(t for r in refs for t in r.targets)),
            ))
    if enumerates and _CHAPEAU_END.search(text):
        found.append(Qualifier(kind="chapeau", span=(0, len(text)), text=text.strip()))
    return sorted(found, key=lambda q: q.span)


def chapeau_ids(in_corpus: Iterable[str]) -> frozenset[str]:
    """The chunk ids that some other chunk hangs off -- the ones a chapeau can govern.

    One pass over the corpus rather than a prefix scan per chunk: the serving path asks this
    of every evidence set, and the scan cost 0.6 ms per cited chunk against 9,961 ids.
    """
    return frozenset(c.rsplit("-", 1)[0] for c in in_corpus if "-" in c.partition("#")[2])


def qualifiers_of(evidence: Mapping[str, str], *,
                  in_corpus: Iterable[str] = ()) -> dict[str, list[Qualifier]]:
    """``qualifiers`` over a whole evidence set, keyed by the version id it came from."""
    parents = in_corpus if isinstance(in_corpus, frozenset) else chapeau_ids(in_corpus)
    out: dict[str, list[Qualifier]] = {}
    for version_id, text in sorted(evidence.items()):
        chunk_id = chunk_id_of(version_id)
        found = qualifiers(text, chunk_id=chunk_id, enumerates=chunk_id in parents)
        if found:
            out[version_id] = found
    return out


# -- did the answer say it? --------------------------------------------------------

#: The lexical marks of a condition in an answer. Not the same list as the triggers: an answer
#: is written in plain language and says "unless", "except", "only", "cannot", "does not
#: apply" where the regulation says "notwithstanding" and "shall not".
#:
#: Connectives only. An earlier list included ``must``, ``when`` and ``upon``, which are marks
#: of obligation and time rather than of a bounded rule -- "the carrier must be notified at
#: least 15 calendar days before the hearing" then counted as acknowledging "unless it is
#: waived in writing by the carrier", which is the exact failure this module exists to catch.
_CUES: dict[str, re.Pattern[str]] = {
    "except": re.compile(r"\bexcept|\bunless\b|\bother than\b|\bexclud|\bdoes not apply\b",
                         re.IGNORECASE),
    "unless": re.compile(r"\bunless\b|\bexcept|\bonly if\b|\bprovided\b|\bas long as\b|"
                         r"\botherwise\b", re.IGNORECASE),
    "subject_to": re.compile(r"\bsubject to\b|\bexcept|\bunless\b|\blimited by\b|"
                             r"\bin accordance\b|\bgoverned by\b|\bconditioned on\b",
                             re.IGNORECASE),
    "notwithstanding": re.compile(r"\bnotwithstanding\b|\bdespite\b|\beven (?:if|though)\b|"
                                  r"\bregardless\b|\boverrides?\b|\binstead of\b|\bexcept",
                                  re.IGNORECASE),
    "proviso": re.compile(r"\bprovided\b|\bas long as\b|\bso long as\b|\bonly if\b|"
                          r"\bconditioned on\b|\bif\b", re.IGNORECASE),
    "only_if": re.compile(r"\bonly\b|\bunless\b|\bexcept|\bif\b", re.IGNORECASE),
    "prohibition": re.compile(r"\bnot\b|\bcannot\b|\bcan't\b|\bno longer\b|\bprohibit|"
                              r"\bbarred\b|\bineligible\b|\bunable\b|\bnever\b", re.IGNORECASE),
    "other_than": re.compile(r"\bother than\b|\bexcept|\bexclud|\bunless\b|\bnot\b|\bonly\b",
                             re.IGNORECASE),
    "chapeau": re.compile(r"\bfollowing\b|\bincluding\b|\beach of\b|\bany of\b|\ball of\b|"
                          r"\bone of\b|\blisted\b|\bconditions?\b|\bcriteri", re.IGNORECASE),
}

#: Share of the qualifier clause's content words the answer must also carry. Swept over
#: {0.15, 0.25, 0.35, 0.50} against three sets in docs/results/eval-006-unstated-conditions.md
#: and 0.25 is the knee: it is the largest value at which none of the 16 hand-written
#: acknowledging answers is falsely flagged, and it already recovers 97.9% of the 480
#: cross-paired cases where the answer discusses a different chunk's condition entirely.
MIN_ACKNOWLEDGEMENT = 0.25


@dataclass(frozen=True)
class UnstatedCondition:
    """A condition in the cited text that the answer does not carry."""

    source: str                     # version id of the cited chunk
    qualifier: Qualifier
    overlap: float                  # share of the clause's content words the answer repeats
    cued: bool                      # the answer carries a marker of this kind of condition

    @property
    def acknowledged(self) -> bool:
        return self.cued and self.overlap >= MIN_ACKNOWLEDGEMENT


def _content(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def acknowledgement(answer: str, qualifier: Qualifier) -> tuple[float, bool]:
    """How much of ``qualifier`` the ``answer`` carries, and whether it is cued as a condition.

    Two signals, and both are needed. Overlap alone says yes to an answer that repeats the
    exception's *subject* while dropping the exception: "Except as provided in paragraph (b),
    an employee must schedule restored leave" and "an employee must schedule restored leave"
    share every content word the clause has that the rule does not. A cue alone says yes to
    an answer that happens to contain the word "not" about something else entirely.

    The words the qualifier shares with the rule it qualifies are not evidence either way, so
    overlap is scored on the clause only -- which is what the character span is for.
    """
    wanted = _content(qualifier.text)
    if not wanted:
        return 0.0, False
    covered = wanted & _content(answer)
    cue = _CUES.get(qualifier.kind)
    return len(covered) / len(wanted), bool(cue and cue.search(answer))


def unstated_conditions(answer: str, evidence: Mapping[str, str], *,
                        in_corpus: Iterable[str] = (),
                        min_overlap: float = MIN_ACKNOWLEDGEMENT) -> list[UnstatedCondition]:
    """Conditions in the cited text that ``answer`` neither states nor gestures at.

    Lexical and structural only, at 10.3 us a pair. The obvious alternative -- ask
    ``verify.entail``'s NLI model whether the answer entails the condition -- was run on the
    same cases and there is no threshold at which it wins on both of them: 0.991 against 0.989
    F1 on the 480 cross-paired cases, and worse on the matched pairs at every threshold that
    reaches parity. It is 1.7 ms a pair warm on a GPU, 200 ms on CPU, and needs a 700 MB
    checkpoint the base install does not have. The model does not earn it here.

    ``bound`` qualifiers are skipped: a stated number is not an unstated condition.
    """
    out: list[UnstatedCondition] = []
    for version_id, found in qualifiers_of(evidence, in_corpus=in_corpus).items():
        for q in found:
            if not q.conditional:
                continue
            overlap, cued = acknowledgement(answer, q)
            if cued and overlap >= min_overlap:
                continue
            out.append(UnstatedCondition(source=version_id, qualifier=q,
                                         overlap=overlap, cued=cued))
    return out
