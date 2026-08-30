"""What the question is actually asking, decided before anything is retrieved.

Until this module existed the user's raw string went straight to FTS5 and to the encoder,
which quietly asserts that every question has the same shape: find text similar to these
words. It does not. "What changed between 2019 and 2023" cannot be answered by one ranked
list at one date; "does this apply to a wage grade employee" is a predicate question whose
correct answer may be that nothing applies; "is there any rule about X" is answered
correctly by returning nothing. A pipeline with no query-understanding stage cannot *fail*
at any of these, because it never attempts them -- the entire class is invisible in the
failure budget, which is the one thing this repository claims to make visible.

Deterministic, and no model on the default path. A query understander that needed a
generation to run would cost twenty seconds against the ~24 ms the rest of retrieval was
measured at, so every decision here is a regex over the query text with the example query
that justifies it recorded beside it -- as a field rather than a comment, because an example
in a comment cannot be asserted against and `tests/test_query.py` asserts every rule still
matches its own.

Nothing here filters. A `QueryPlan` is a set of *proposals* -- an as-of date, a scope, a
rewritten query, a multi-hop flag -- and the caller decides what to do with them. Two
consequences are deliberate:

- An unmatched query is reported as ``lookup`` at low confidence, not guessed at. A
  misrouted query is worse than an unrouted one, because an unrouted one still gets the
  ordinary hybrid answer.
- A scope facet is proposed only on an unambiguous signal. An unrequested scope filter
  silently removes 41% of the corpus (see `scope.Scope.of`), and returning no facet costs
  only breadth.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import StrEnum

from .scope import GOVERNMENT_WIDE, Scope, known_values


class QueryKind(StrEnum):
    """What shape of answer the question wants.

    ``TEMPORAL_COMPARE`` is the one the rest of the system cannot yet serve: `Retriever` and
    the API take a single ``as_of``, so a two-date question has no expressible form. Naming
    it here is the first step to answering it; until then it is a plan that says, precisely,
    what was asked and could not be done.
    """

    LOOKUP = "lookup"
    TEMPORAL_POINT = "temporal_point"
    TEMPORAL_COMPARE = "temporal_compare"
    APPLICABILITY = "applicability"
    AGGREGATE = "aggregate"
    ABSENCE = "absence"


#: Confidence reported when nothing matched. Deliberately low and deliberately not zero: it
#: means "no route was identified", not "this is a lookup", and a caller that thresholds on
#: confidence should see the difference between a recognised lookup and an unrecognised one.
LOOKUP_CONFIDENCE = 0.35
MAX_CONFIDENCE = 0.95

#: Below this many content terms a rewritten query cannot locate anything, so the rewriter
#: gives back what it stripped rather than shipping a one-word query. Two, not three:
#: "annual leave" and "credit hours" are complete questions in this corpus.
MIN_RETRIEVAL_TERMS = 2

#: A follow-up turn with fewer content terms than this is treated as elliptical and given the
#: previous turn's subject. Merging costs dilution, which retrieval recovers from; dropping
#: the subject answers a different question, which it does not.
FOLLOW_UP_TERMS = 3


# -- dates --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DateHint:
    """One date the query named, and the exact phrase it was read from.

    ``text`` is kept verbatim so the rewriter can remove precisely what the extractor
    consumed. Leaving "in 2019" in the retrieval query makes BM25 match the token "2019" in
    every unrelated paragraph that happens to cite a 2019 Federal Register notice.

    ``is_now`` marks "currently" / "the current rules": a date the query implied rather than
    named. It sets an as-of, because a trace that records the date it answered at is worth
    more than one that records None, but it does not make the question a temporal one.
    """

    iso: str
    text: str
    granularity: str = "day"                # "day" | "month" | "year"
    is_now: bool = False


_MONTH_NAMES = ("january", "february", "march", "april", "may", "june", "july",
                "august", "september", "october", "november", "december")
_MONTH_NUMBER = {name[:3]: i for i, name in enumerate(_MONTH_NAMES, start=1)}
_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?"
          r"|aug(?:ust)?|sept(?:ember)?|sep|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")

#: A date as a person writes one. Non-capturing throughout so the surrounding patterns can
#: each grab a whole atom in a single group and hand the substring back to `_parse_atom`;
#: Python allows a group name only once per pattern, and a range needs two atoms.
_ATOM = (r"(?:\d{4}-\d{2}-\d{2}"
         rf"|{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}"
         rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH},?\s+\d{{4}}"
         rf"|{_MONTH}\s+\d{{4}}"
         r"|(?:19|20)\d{2})")

_ISO_ATOM = re.compile(r"(\d{4})-(\d{2})-(\d{2})$")
_MDY_ATOM = re.compile(rf"({_MONTH})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})$", re.I)
_DMY_ATOM = re.compile(rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH}),?\s+(\d{{4}})$", re.I)
_MY_ATOM = re.compile(rf"({_MONTH})\s+(\d{{4}})$", re.I)
_Y_ATOM = re.compile(r"((?:19|20)\d{2})$")

#: An amendment year is almost always written with the noun that names it, and the noun has
#: to be consumed too or "amendment" survives into the retrieval query as a content term.
_AMENDMENT_NOUN = r"(?:\s+(?:amendments?|changes?|revisions?|rules?|version|update))?"

_RANGE_RE = re.compile(
    rf"\b(?:between|from)\s+({_ATOM})\s+(?:and|to|through|until)\s+({_ATOM})", re.I)
_BEFORE_RE = re.compile(
    rf"\b(?:before|prior\s+to|preceding|up\s+to)\s+(?:the\s+)?({_ATOM}){_AMENDMENT_NOUN}", re.I)
_AFTER_RE = re.compile(
    rf"\b(?:after|since|following)\s+(?:the\s+)?({_ATOM}){_AMENDMENT_NOUN}", re.I)
_ASOF_RE = re.compile(
    rf"\b(?:as\s+of|as\s+at|effective|back\s+in|during|on|in|by)\s+({_ATOM})", re.I)
_BARE_RE = re.compile(rf"\b(?:the\s+)?({_ATOM}){_AMENDMENT_NOUN}", re.I)
_RELATIVE_RE = re.compile(
    r"\b(?:as\s+of\s+today|right\s+now|at\s+present|currently|today|now"
    r"|the\s+current\s+(?:rules?|regulations?|version)|this\s+year|last\s+year)\b", re.I)


def _parse_atom(atom: str) -> tuple[int, int | None, int | None] | None:
    """(year, month, day) for one date atom; month/day are None where unnamed."""
    text = atom.strip()
    if (m := _ISO_ATOM.match(text)):
        return int(m[1]), int(m[2]), int(m[3])
    if (m := _MDY_ATOM.match(text)):
        return int(m[3]), _MONTH_NUMBER[m[1][:3].lower()], int(m[2])
    if (m := _DMY_ATOM.match(text)):
        return int(m[3]), _MONTH_NUMBER[m[2][:3].lower()], int(m[1])
    if (m := _MY_ATOM.match(text)):
        return int(m[2]), _MONTH_NUMBER[m[1][:3].lower()], None
    if (m := _Y_ATOM.match(text)):
        return int(m[1]), None, None
    return None


def _period(year: int, month: int | None, day: int | None) -> tuple[date, date]:
    """First and last day of the period an atom names. Raises for an unreal calendar date."""
    if day is not None and month is not None:
        one = date(year, month, day)
        return one, one
    if month is not None:
        return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])
    return date(year, 1, 1), date(year, 12, 31)


def _resolve(atom: str, *, edge: str, today: date) -> tuple[str, str] | None:
    """One atom as an ISO date, or None if it names no real day.

    Every partial date resolves to the **last** day of the period it names: 2019 to
    2019-12-31, June 2021 to 2021-06-30. Uniform, and it is what "as of <period>" means in
    legal usage -- as of the close of that period. ``edge="before"`` takes the day before the
    period instead, so "before the 2020 amendment" is the law as it stood entering 2020.

    Future dates are clamped to ``today``. The corpus cannot know a text that has not been
    published, and an as-of the store would answer from today's rows anyway; clamping makes
    the date recorded on the trace true instead of aspirational.
    """
    parsed = _parse_atom(atom)
    if parsed is None:
        return None
    year, month, day = parsed
    try:
        first, last = _period(year, month, day)
    except ValueError:
        return None                         # 2021-02-30 and friends
    picked = first - timedelta(days=1) if edge == "before" else last
    if picked > today:
        picked = today
    grain = "day" if day is not None else ("month" if month is not None else "year")
    return picked.isoformat(), grain


def extract_dates(text: str, *, today: date | None = None) -> list[DateHint]:
    """Every date the query names, in the order it named them.

    Ranges first, then the qualified forms, then bare years, with each match reserving its
    span so a later pattern cannot re-read the same characters. Without that, "between 2019
    and 2023" yields the range *and* two bare years, and the compare pair silently becomes
    four dates.
    """
    now = today or date.today()
    taken: list[tuple[int, int]] = []
    found: list[tuple[int, DateHint]] = []

    def free(span: tuple[int, int]) -> bool:
        return not any(span[0] < end and start < span[1] for start, end in taken)

    def claim(match: re.Match[str], hints: list[DateHint]) -> None:
        if not hints or not free(match.span()):
            return
        taken.append(match.span())
        found.extend((match.start(), h) for h in hints)

    for m in _RANGE_RE.finditer(text):
        a = _resolve(m[1], edge="end", today=now)
        b = _resolve(m[2], edge="end", today=now)
        claim(m, [DateHint(a[0], m[0], a[1]), DateHint(b[0], m[0], b[1])] if a and b else [])

    for pattern, edge in ((_BEFORE_RE, "before"), (_AFTER_RE, "end"), (_ASOF_RE, "end")):
        for m in pattern.finditer(text):
            got = _resolve(m[1], edge=edge, today=now)
            claim(m, [DateHint(got[0], m[0], got[1])] if got else [])

    for m in _RELATIVE_RE.finditer(text):
        phrase = m[0].lower()
        if phrase == "last year":
            iso, grain = date(now.year - 1, 12, 31).isoformat(), "year"
        elif phrase == "this year":
            # The year has not finished, so its "last day" is in the future; the honest
            # reading of "this year" is the text in force as this request is answered.
            iso, grain = now.isoformat(), "year"
        else:
            iso, grain = now.isoformat(), "day"
        claim(m, [DateHint(iso, m[0], grain, is_now=phrase not in ("last year",))])

    for m in _BARE_RE.finditer(text):
        got = _resolve(m[1], edge="end", today=now)
        claim(m, [DateHint(got[0], m[0], got[1])] if got else [])

    seen: dict[str, None] = {}
    hints: list[DateHint] = []
    for _, hint in sorted(found, key=lambda pair: pair[0]):
        if hint.iso in seen:
            continue
        seen[hint.iso] = None
        hints.append(hint)
    return hints


# -- scope --------------------------------------------------------------------------

#: Wording that frames a facet as a description of the asker rather than of the subject, and
#: the article that follows it. Consumed with the facet phrase so that stripping "a GS
#: employee" does not leave "for a" behind in the retrieval query.
_FRAME = r"(?:\b(?:as|for|to|of|under)\s+)?(?:\b(?:an?|the)\s+)?"
_WHO = r"(?:\s+(?:employees?|workers?|positions?|appointees?|members?|staff|status))?"


@dataclass(frozen=True)
class ScopeRule:
    name: str
    pattern: re.Pattern[str]
    #: facet -> value pairs, as pairs so the rule table stays hashable and diffable.
    facets: tuple[tuple[str, str], ...]
    example: str


#: Read straight off `scope.FACET_VALUES`; no rule may propose a value that is not declared
#: there, and `extract_scope` re-checks rather than trusting this table.
SCOPE_RULES: tuple[ScopeRule, ...] = (
    ScopeRule(
        "gs",
        re.compile(_FRAME + r"\b(?:general\s+schedule|GS(?:[-\s]?\d{1,2})?)\b" + _WHO, re.I),
        (("pay_system", "GS"),),
        "Does the waiting period apply to a GS employee?"),
    ScopeRule(
        "fws",
        re.compile(_FRAME + r"\b(?:federal\s+wage\s+system|wage[-\s]grade|prevailing\s+rate"
                            r"|WG[-\s]?\d{1,2}|FWS)\b" + _WHO, re.I),
        (("pay_system", "FWS"),),
        "What is the pay cap for a wage grade employee?"),
    # An SES member is in the SES pay system *and* the SES service, so one phrase settles two
    # facets. Both are load-bearing: pay_system=SES keeps parts 511/531/532 out, and
    # service=SES keeps out 315/316/337, which govern the competitive service only.
    ScopeRule(
        "ses",
        re.compile(_FRAME + r"\b(?:senior\s+executive\s+service|SES)\b" + _WHO, re.I),
        (("pay_system", "SES"), ("service", "SES")),
        "How is SES performance appraised?"),
    ScopeRule(
        "competitive",
        re.compile(_FRAME + r"\bcompetitive\s+service\b" + _WHO, re.I),
        (("service", "competitive"),),
        "Who may be appointed in the competitive service?"),
    ScopeRule(
        "excepted",
        re.compile(_FRAME + r"\bexcepted\s+(?:service|appointments?)\b" + _WHO, re.I),
        (("service", "excepted"),),
        "Does probation apply in the excepted service?"),
)


def extract_scope(text: str) -> tuple[Scope, dict[str, str]]:
    """The profile the query asks under, and the phrase each facet was read from.

    Two refusals, both deliberate:

    - A facet named with two different values is dropped entirely. "How does GS pay compare
      with wage grade pay?" is a comparison, and answering it under either pay system is
      answering a different question -- with five parts, 41% of the corpus, removed first.
    - A value the rule table proposes but `scope.FACET_VALUES` does not declare is dropped
      rather than passed to `Scope.of`, which would raise. A typo in this table must not
      become a 500 on a request.

    The returned phrases are only those of surviving facets, so a conflicting query keeps its
    wording in the retrieval text, where it is topical rather than a filter.
    """
    proposed: dict[str, set[str]] = {}
    phrases: dict[str, str] = {}
    for rule in SCOPE_RULES:
        match = rule.pattern.search(text)
        if match is None:
            continue
        for facet, value in rule.facets:
            proposed.setdefault(facet, set()).add(value)
            phrases.setdefault(facet, match[0].strip())

    facets: dict[str, str] = {}
    for facet, values in proposed.items():
        value = next(iter(values))
        if len(values) != 1 or value not in known_values(facet):
            phrases.pop(facet, None)
            continue
        facets[facet] = value
    return Scope.of(**facets), phrases


# -- citations ----------------------------------------------------------------------

#: Part numbers in this corpus are three digits (300-890). Requiring exactly three keeps
#: "1.5" and a version string out, and keeps the year in "2019.06" out: there is no word
#: boundary before the last three digits of a four-digit number.
_CFR_PREFIX = r"(?:\btitle\s+)?\d{1,2}\s*C\.?\s*F\.?\s*R\.?\s*"
_SECTION_WORD = r"(?:§{1,2}\s*|\bsec(?:tion|\.)?\s*)"
_CITATION = re.compile(
    rf"(?:{_CFR_PREFIX})?(?:{_SECTION_WORD})?\b(\d{{3}})\.(\d{{1,4}})\b(?:\([a-z0-9]{{1,3}}\))*",
    re.I)


def find_citations(text: str) -> list[str]:
    """Every section this query cites, as the ``630.306`` form the store keys on."""
    seen: dict[str, None] = {}
    for m in _CITATION.finditer(text):
        seen[f"{m[1]}.{m[2]}"] = None
    return list(seen)


def normalise_citations(text: str) -> str:
    """Rewrite every citation form to the bare section number.

    "5 CFR 630.306", "5 C.F.R. § 630.306", "§630.306", "section 630.306" and "630.306" are
    one address written five ways, and FTS5 sees five different token sequences: the first
    contributes the useless high-frequency tokens "5" and "CFR", and the section symbol is
    bare punctuation, which `hybrid.fts_query` turns into nothing at all. Paragraph letters
    are dropped with them -- "(a)" tokenises to "a" and matches everything.
    """
    return _CITATION.sub(lambda m: f"{m[1]}.{m[2]}", text)


# -- rewriting ----------------------------------------------------------------------

#: Expanded *beside* the acronym rather than in place of it. eCFR spells out "reduction in
#: force" in the operative text while defining and then using "RIF" as a term of art, so
#: replacing either form loses the paragraphs that use the other.
ABBREVIATIONS: dict[str, str] = {
    "RIF": "reduction in force",
    "WGI": "within-grade increase",
    "CTAP": "career transition assistance program",
    "ICTAP": "interagency career transition assistance program",
    "FEHB": "federal employees health benefits",
    "FLSA": "fair labor standards act",
    "SES": "senior executive service",
    "USERRA": "uniformed services employment and reemployment rights act",
}

_LEAD_FORMS = (
    r"please",
    r"(?:so|and|but|ok|okay)\s*,?",
    r"i\s+(?:want|need|would\s+like)\s+to\s+know(?:\s+(?:if|whether|what))?",
    r"(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:tell\s+me|explain|show\s+me|list)",
    r"tell\s+me(?:\s+about)?",
    r"what\s+happens\s+(?:if|when)",
    r"what\s+about",
    r"what(?:\s+has|\s+have)?\s+changed",
    r"what(?:'s|’s|\s+is|\s+are|\s+was|\s+were|\s+does|\s+do|\s+did)",
    r"how\s+(?:long|much|many|often)(?:\s+(?:is|are|do|does|can|could|may|must))?",
    r"how\s+(?:do|does|can|could|should|would)(?:\s+(?:i|we|you|a|an|the))?",
    r"(?:am|are)\s+i\s+(?:entitled\s+to|eligible\s+for|allowed\s+to|required\s+to|subject\s+to)",
    r"am\s+i",
    r"is\s+there(?:\s+(?:any|a|an))?",
    r"are\s+there(?:\s+any)?",
    r"(?:can|may|could|should|must|do|does|did|is|are|was|were|will|would)"
    r"\s+(?:i|we|you|they|a|an|the|this|that|there)",
    r"(?:any|the)\s+(?:rules?|regulations?|provisions?|requirements?)"
    r"\s+(?:about|on|for|regarding|covering|governing)",
    r"list(?:\s+(?:all|every|each))?(?:\s+(?:of\s+)?the)?",
    r"which\s+(?:sections?|parts?|subparts?|regulations?)",
    r"who\s+(?:is|are)",
    # The bare interrogatives, last so the multi-word forms above claim their whole phrase
    # first. None of the five is ever a content term in a regulation.
    r"(?:what|how|why|when|where|who)",
    r"(?:for|about|regarding|concerning|on|in|of|with|to)",
    r"(?:the|a|an|any|my)",
)
_LEAD = re.compile(r"^\s*(?:" + "|".join(_LEAD_FORMS) + r")(?=\s|$|[?.,;])\s*", re.I)
_TRAIL = re.compile(
    r"(?:\s+(?:apply|applies|to|for|in|on|of|about|regarding|and|or|the|a|an"
    r"|me|us|my|it|that|this|they|them))+\s*$", re.I)

#: Auxiliaries that survive the leading strip in the middle of a sentence and carry no
#: topical weight here. ``may``, ``must``, ``shall`` and ``can`` are deliberately absent:
#: they are operative words in regulation ("an agency may"), not filler.
_AUX = re.compile(r"\b(?:does|do|did|is|are|was|were|been|am)\b", re.I)

_FILLER = frozenset(
    "a an the of for to in on at by and or is are was were do does did my me i it that "
    "this these those there".split())
_TOKEN = re.compile(r"[A-Za-z0-9][\w.’'-]*")
_MAX_STRIP_ROUNDS = 6


def _content_tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text) if t.lower() not in _FILLER]


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text.strip(" \t\r\n?!.,;:-")


def strip_scaffolding(text: str) -> str:
    """Remove interrogative framing from the front and dangling function words from the end.

    Iterated, because the forms nest: "So, what is the ..." sheds three layers. It stops the
    moment a round would empty the string -- an empty retrieval query is a silent recall
    failure, and the whole point of the ladder in `rewrite` is that stripping is reversible.
    """
    out = text.strip()
    for _ in range(_MAX_STRIP_ROUNDS):
        new = _TRAIL.sub("", _LEAD.sub("", out, count=1), count=1).strip(" \t?.!,;:")
        if not new or new == out:
            break
        out = new
    return out


def abbreviations_in(text: str) -> dict[str, str]:
    """The domain abbreviations present in ``text``, with what each stands for."""
    return {a: e for a, e in ABBREVIATIONS.items() if re.search(rf"\b{a}\b", text, re.I)}


def expand_abbreviations(text: str) -> str:
    """Add each abbreviation's expansion beside its first occurrence.

    Skipped where the expansion is already written out: a query saying "reduction in force
    (RIF)" needs nothing added, and adding it anyway would spend three of the 64 tokens
    `hybrid.MAX_QUERY_TOKENS` allows on a duplicate that contributes no new postings.
    """
    for abbr, expansion in sorted(ABBREVIATIONS.items(), key=lambda kv: -len(kv[0])):
        pattern = re.compile(rf"\b{abbr}\b", re.I)
        if expansion.lower() in text.lower() or not pattern.search(text):
            continue
        text = pattern.sub(f"{abbr} {expansion}", text, count=1)
    return text


def _rewrite_once(text: str, phrases: Sequence[str]) -> str:
    for phrase in sorted({p for p in phrases if p}, key=len, reverse=True):
        text = re.sub(re.escape(phrase), " ", text, flags=re.I)
    text = strip_scaffolding(text)
    text = strip_scaffolding(_AUX.sub(" ", text))
    return _tidy(expand_abbreviations(text))


def rewrite(query: str, *, dates: Sequence[str] = (), scope_phrases: Sequence[str] = ()) -> str:
    """The text retrieval should actually see.

    Ordered so that each step works on what the last one left: citations are normalised
    first (before token-level stripping can break "5 CFR" apart), then the phrases already
    captured as structured fields are removed, then the interrogative framing, then the
    abbreviations are expanded.

    The ladder is the safety net. Dropping the scope wording is right when it describes the
    asker ("as a GS employee") and wrong when it *is* the subject ("what is prevailing rate
    pay") -- and no deterministic rule separates those two reliably. So the rewriter tries
    the fullest strip first and gives back what it removed, scope before dates, until at
    least `MIN_RETRIEVAL_TERMS` content terms survive. Giving back the wording costs a little
    ranking noise; keeping a one-word query costs the answer. The scope *predicate* is
    unaffected either way -- only the lexical text changes.
    """
    text = normalise_citations(query)
    attempt = ""
    for phrases in (tuple(dates) + tuple(scope_phrases), tuple(dates), ()):
        attempt = _rewrite_once(text, phrases)
        if len(_content_tokens(attempt)) >= MIN_RETRIEVAL_TERMS:
            return attempt
    # Nothing survived stripping. The user's own words rank worse than a clean query and
    # infinitely better than an empty one.
    return attempt or _tidy(text) or query.strip()


# -- classification -----------------------------------------------------------------


@dataclass(frozen=True)
class ClassifierRule:
    kind: QueryKind
    name: str
    pattern: re.Pattern[str]
    confidence: float
    #: A real query this rule exists for. A field rather than a comment so the test suite can
    #: assert the rule still matches it; a comment that has drifted from its regex is worse
    #: than no comment, because it is believed.
    example: str
    #: False for a rule that may raise confidence but may not choose the class alone. Past
    #: tense ("what did the rule say") is a phrasing habit as often as it is a question about
    #: the past, and routing on it alone sent ordinary lookups to a date that was never named.
    standalone: bool = True


CLASSIFIER_RULES: tuple[ClassifierRule, ...] = (
    # -- temporal_compare: two states of the law, which one as-of cannot express -----
    ClassifierRule(QueryKind.TEMPORAL_COMPARE, "what-changed",
                   re.compile(r"\bwhat\s+(?:has\s+)?changed\b", re.I), 0.90,
                   "What changed in the annual leave rules between 2019 and 2023?"),
    ClassifierRule(QueryKind.TEMPORAL_COMPARE, "how-changed",
                   re.compile(r"\bhow\s+(?:has|have|did|was|were)\b.*\bchang", re.I), 0.85,
                   "How has the probationary period changed since 2017?"),
    ClassifierRule(QueryKind.TEMPORAL_COMPARE, "difference-between",
                   re.compile(r"\bdifferences?\s+between\b", re.I), 0.85,
                   "What is the difference between the 2019 and 2021 carryover limits?"),
    ClassifierRule(QueryKind.TEMPORAL_COMPARE, "before-and-after",
                   re.compile(r"\bbefore\s+and\s+after\b", re.I), 0.85,
                   "What were the retention rules before and after the 2020 amendment?"),
    ClassifierRule(QueryKind.TEMPORAL_COMPARE, "compare",
                   re.compile(r"\b(?:compare[ds]?|versus|vs\.?)\b", re.I), 0.70,
                   "Compare the RIF retention rules in 2018 and 2024."),
    ClassifierRule(QueryKind.TEMPORAL_COMPARE, "used-to",
                   re.compile(r"\bused\s+to\b", re.I), 0.70,
                   "What did the rule used to say about credit hours?"),

    # -- absence: the correct answer may be that nothing governs this ---------------
    ClassifierRule(QueryKind.ABSENCE, "is-there-any",
                   re.compile(r"\bis\s+there\s+(?:any|a|an)\b", re.I), 0.75,
                   "Is there any rule about remote work equipment reimbursement?"),
    ClassifierRule(QueryKind.ABSENCE, "are-there-any",
                   re.compile(r"\bare\s+there\s+any\b", re.I), 0.75,
                   "Are there any provisions covering telework stipends?"),
    ClassifierRule(QueryKind.ABSENCE, "is-there-anything",
                   re.compile(r"\bis\s+there\s+anything\b", re.I), 0.80,
                   "Is there anything in the regulations about compressed schedules?"),
    ClassifierRule(QueryKind.ABSENCE, "does-any-regulation",
                   re.compile(r"\b(?:does|do)\s+any\s+(?:rules?|regulations?|provisions?)\b",
                              re.I), 0.75,
                   "Does any regulation address commuting costs?"),
    ClassifierRule(QueryKind.ABSENCE, "any-rules-about",
                   re.compile(r"\bany\s+(?:rules?|regulations?|provisions?)\s+"
                              r"(?:about|on|for|covering|governing)\b", re.I), 0.70,
                   "Any rules about wearing a uniform on duty?"),

    # -- applicability: a predicate about the asker, not a similarity question ------
    ClassifierRule(QueryKind.APPLICABILITY, "does-this-apply",
                   re.compile(r"\b(?:does|do)\s+(?:this|that|it|these|those)\s+apply\b", re.I),
                   0.90, "Does this apply to me?"),
    ClassifierRule(QueryKind.APPLICABILITY, "apply-to",
                   re.compile(r"\bappl(?:y|ies)\s+to\s+(?:me|my|us|a|an|the)\b", re.I), 0.80,
                   "Does the within-grade increase waiting period apply to "
                   "a wage grade employee?"),
    ClassifierRule(QueryKind.APPLICABILITY, "am-i",
                   re.compile(r"\bam\s+i\s+(?:eligible|entitled|covered|subject|required"
                              r"|allowed)\b", re.I), 0.85,
                   "Am I entitled to a within-grade increase?"),
    ClassifierRule(QueryKind.APPLICABILITY, "do-i-qualify",
                   re.compile(r"\b(?:do|am)\s+i\s+qualif", re.I), 0.85,
                   "Do I qualify for career tenure?"),
    ClassifierRule(QueryKind.APPLICABILITY, "eligible-for",
                   re.compile(r"\b(?:eligible|entitled)\s+(?:for|to)\b", re.I), 0.60,
                   "Is a term employee eligible for a within-grade increase?"),
    ClassifierRule(QueryKind.APPLICABILITY, "covered-by",
                   re.compile(r"\b(?:covered|governed)\s+by\b", re.I), 0.60,
                   "Which employees are covered by the FLSA?"),

    # -- aggregate: a set question, which one ranked list of 8 cannot answer --------
    ClassifierRule(QueryKind.AGGREGATE, "list-all",
                   re.compile(r"\blist\s+(?:all|the|every|each)\b", re.I), 0.85,
                   "List all the parts that govern reduction in force."),
    ClassifierRule(QueryKind.AGGREGATE, "how-many",
                   re.compile(r"\bhow\s+many\s+(?:sections?|parts?|subparts?|rules?"
                              r"|regulations?|provisions?|paragraphs?|types?|kinds?"
                              r"|categories)\b", re.I), 0.85,
                   "How many sections cover probationary periods?"),
    ClassifierRule(QueryKind.AGGREGATE, "which-sections",
                   re.compile(r"\bwhich\s+(?:sections?|parts?|subparts?|regulations?)\b", re.I),
                   0.80, "Which sections mention within-grade increases?"),
    ClassifierRule(QueryKind.AGGREGATE, "what-are-all",
                   re.compile(r"\bwhat\s+are\s+all\s+(?:of\s+)?the\b", re.I), 0.80,
                   "What are all of the exceptions to the annual leave ceiling?"),
    ClassifierRule(QueryKind.AGGREGATE, "enumerate",
                   re.compile(r"\b(?:enumerate|count\s+the)\b", re.I), 0.70,
                   "Enumerate the reduction in force retention factors."),

    # -- temporal_point: one state of the law, at a date the query names ------------
    ClassifierRule(QueryKind.TEMPORAL_POINT, "as-of",
                   re.compile(r"\bas\s+of\b", re.I), 0.85,
                   "What was the annual leave carryover limit as of June 2021?"),
    ClassifierRule(QueryKind.TEMPORAL_POINT, "at-the-time",
                   re.compile(r"\bat\s+the\s+time\b", re.I), 0.75,
                   "At the time of the 2020 amendment, what was the accrual rate?"),
    ClassifierRule(QueryKind.TEMPORAL_POINT, "back-in",
                   re.compile(r"\bback\s+in\b", re.I), 0.75,
                   "Back in 2018, how much leave could be carried over?"),
    ClassifierRule(QueryKind.TEMPORAL_POINT, "past-tense",
                   re.compile(r"\bwhat\s+(?:did|was|were)\b", re.I), 0.55,
                   "What did the rule say in 2019?", standalone=False),
)

#: Checked in this order, and the first class with a standalone signal wins. The order is the
#: cost of being wrong: a two-date question served at one date is answered confidently about
#: the wrong law, whereas a temporal_point question misread as a lookup still retrieves at
#: the caller's default date. ``lookup`` is last because it is the absence of a route.
_PRECEDENCE = (QueryKind.TEMPORAL_COMPARE, QueryKind.ABSENCE, QueryKind.APPLICABILITY,
               QueryKind.AGGREGATE, QueryKind.TEMPORAL_POINT)


@dataclass(frozen=True)
class Classification:
    kind: QueryKind
    confidence: float
    #: Every pattern that fired, including ones the precedence order passed over. An autopsy
    #: of a misroute needs to see what nearly won, not only what did.
    signals: list[str] = field(default_factory=list)


def classify(text: str, *, dates: Sequence[DateHint] = ()) -> Classification:
    """Label the query, defaulting to ``lookup`` rather than to the nearest guess.

    The dates are a signal in their own right and not merely a filter: two dates the query
    named *is* a comparison however it is phrased, and one named date is a question about
    that date even when no temporal wording appears ("the 2019 GS pay cap"). Dates the query
    only implied (``is_now``) are excluded -- asking about now is not a temporal question.
    """
    matched: dict[QueryKind, list[ClassifierRule]] = {}
    for rule in CLASSIFIER_RULES:
        if rule.pattern.search(text):
            matched.setdefault(rule.kind, []).append(rule)

    explicit = [d for d in dates if not d.is_now]
    from_dates: dict[QueryKind, tuple[float, str]] = {}
    if len(explicit) >= 2:
        from_dates[QueryKind.TEMPORAL_COMPARE] = (0.70, "two-dates")
    elif explicit:
        from_dates[QueryKind.TEMPORAL_POINT] = (0.70, "one-date")

    for kind in _PRECEDENCE:
        rules = matched.get(kind, [])
        extra = from_dates.get(kind)
        if not any(r.standalone for r in rules) and extra is None:
            continue
        weights = [r.confidence for r in rules] + ([extra[0]] if extra else [])
        names = [f"{kind}:{r.name}" for r in rules] + ([f"{kind}:{extra[1]}"] if extra else [])
        # Two independent patterns agreeing is more than one of them; the bump is small
        # because they are correlated by construction -- both were written for this class.
        confidence = min(MAX_CONFIDENCE, max(weights) + 0.05 * (len(weights) - 1))
        return Classification(kind, round(confidence, 3), names)

    near_misses = [f"{k}:{r.name}" for k, rs in matched.items() for r in rs]
    return Classification(QueryKind.LOOKUP, LOOKUP_CONFIDENCE, near_misses)


# -- multi-hop ----------------------------------------------------------------------

#: 15.7% of in-force paragraphs cite another section, so a query whose answer sits behind one
#: of those pointers is common enough to be worth naming. Every pattern here is wording that
#: the *question* uses when the answer is a chain: a definition that lives elsewhere, an
#: exception that lives elsewhere, a condition imported from elsewhere.
_MULTI_HOP = (
    ("as-defined-in", r"\bas\s+defined\s+in\b"),
    ("as-provided-in", r"\bas\s+(?:provided|described|set\s+forth|specified)\s+in\b"),
    ("pursuant-to", r"\bpursuant\s+to\b"),
    ("subject-to", r"\bsubject\s+to\b"),
    ("refers-to", r"\brefer(?:s|red|ring)?\s+to\b"),
    ("cross-reference", r"\bcross[-\s]?reference"),
    ("exceptions", r"\bexcept(?:ion|ions)\s+(?:to|from|in|under)\b"),
    ("definition-of", r"\bdefinition\s+of\b"),
    ("what-does-x-mean", r"\bwhat\s+(?:does|do)\b.{0,40}\bmean\b"),
    ("for-purposes-of", r"\bfor\s+(?:the\s+)?purposes\s+of\b"),
)
_MULTI_HOP_RULES = tuple((name, re.compile(p, re.I)) for name, p in _MULTI_HOP)


def likely_multi_hop(text: str, *, citations: Sequence[str] = ()) -> bool:
    """Whether the answer probably sits behind a cross-reference.

    The hop is *not* taken here. Following it means a second retrieval round with its own
    latency and its own failure mode, and a flag a later stage can act on is worth more than
    a hop this stage cannot attribute in the trace. Two citations in one query count as well:
    a question naming two sections is asking how they relate.
    """
    if len(set(citations)) >= 2:
        return True
    return any(pattern.search(text) for _, pattern in _MULTI_HOP_RULES)


# -- the plan -----------------------------------------------------------------------


@dataclass(frozen=True)
class QueryPlan:
    """Everything this stage worked out, and nothing it did about it.

    ``as_of`` is None when the query named no date -- the caller's default stands. For a
    ``temporal_compare`` plan it is the *later* of ``compare_dates``, so a caller wired to
    the single-``as_of`` API still gets the more useful half of the question rather than an
    arbitrary one, and the trace records that the other half was asked for and dropped.
    """

    raw: str
    kind: QueryKind
    confidence: float
    retrieval_query: str
    as_of: str | None = None
    compare_dates: tuple[str, str] | None = None
    scope: Scope = GOVERNMENT_WIDE
    #: facet -> the phrase it was read from. Empty for an inferred-nothing plan, which is the
    #: common and correct case.
    scope_evidence: dict[str, str] = field(default_factory=dict)
    dates: list[DateHint] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    abbreviations: dict[str, str] = field(default_factory=dict)
    needs_multi_hop: bool = False
    signals: list[str] = field(default_factory=list)
    #: What came from earlier turns rather than from this one: "subject", "scope", "as_of".
    #: Recorded because an inherited filter is invisible in the question the user typed, and
    #: a wrong inheritance is the failure mode that answers for the wrong person or year.
    inherited: list[str] = field(default_factory=list)

    def as_of_or(self, default: str) -> str:
        """The date to retrieve at: what the query asked for, else the caller's default."""
        return self.as_of or default


_ELLIPTICAL = re.compile(
    r"^\s*(?:what\s+about|how\s+about|and|but|what\s+if|same\s+(?:for|question))\b", re.I)
_PRONOUNS = re.compile(r"\b(?:it|its|that|this|they|them|those|these|the\s+same)\b", re.I)


def _dedupe_tokens(text: str) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for token in text.split():
        key = token.lower().strip(".,;:")
        if key in seen:
            continue
        seen.add(key)
        out.append(token)
    return " ".join(out)


def _carry_subject(retrieval: str, raw: str, previous: QueryPlan) -> tuple[str, bool]:
    """Give an elliptical follow-up the previous turn's subject.

    Three triggers, any one of which is enough: an opening that announces a follow-up ("what
    about ..."), a pronoun with no antecedent in this turn, or too few content terms to stand
    alone. The merge is biased towards over-inclusion on purpose -- extra terms are diluted
    by rank fusion, whereas a follow-up retrieved without its subject is a confident answer to
    a question nobody asked.
    """
    if not previous.retrieval_query:
        return retrieval, False
    elliptical = (_ELLIPTICAL.match(raw) is not None
                  or _PRONOUNS.search(retrieval) is not None
                  or len(_content_tokens(retrieval)) < FOLLOW_UP_TERMS)
    if not elliptical:
        return retrieval, False
    own = _tidy(_PRONOUNS.sub(" ", retrieval))
    return _dedupe_tokens(_tidy(f"{previous.retrieval_query} {own}")), True


def plan_query(query: str, *, history: Sequence[QueryPlan] = (),
               today: date | None = None) -> QueryPlan:
    """Understand one question. Deterministic, offline, and cheap enough to always run.

    ``history`` is the previous turns' plans, oldest first; only the last is consulted.
    ``today`` exists so that "currently" and "last year" are reproducible in a test and in a
    replay -- a stage whose output depends on the wall clock cannot be replayed at all.
    """
    now = today or date.today()
    raw = query.strip()
    previous = history[-1] if history else None

    dates = extract_dates(raw, today=now)
    scope, evidence = extract_scope(raw)
    citations = find_citations(raw)
    classification = classify(raw, dates=dates)
    retrieval = rewrite(raw, dates=[h.text for h in dates],
                        scope_phrases=list(evidence.values()))

    explicit = [h for h in dates if not h.is_now]
    as_of = dates[-1].iso if dates else None
    compare = ((explicit[0].iso, explicit[-1].iso)
               if classification.kind is QueryKind.TEMPORAL_COMPARE and len(explicit) >= 2
               else None)
    if compare is not None:
        as_of = compare[1]

    inherited: list[str] = []
    if previous is not None:
        retrieval, carried = _carry_subject(retrieval, raw, previous)
        if carried:
            inherited.append("subject")
        # Both filters are inherited only where this turn is silent about them. The follow-up
        # always wins when it speaks: "what about wage grade employees?" after a GS question
        # is a new profile, not a violation of the old one.
        if not scope.facets and previous.scope.facets:
            scope, evidence = previous.scope, dict(previous.scope_evidence)
            inherited.append("scope")
        if not dates and previous.as_of:
            as_of = previous.as_of
            inherited.append("as_of")

    signals = list(classification.signals)
    signals += [f"scope:{facet}={value}" for facet, value in sorted(scope.facets.items())]
    signals += [f"date:{h.iso}" for h in dates]
    multi_hop = likely_multi_hop(raw, citations=citations)
    if multi_hop:
        signals.append("multi-hop")

    return QueryPlan(
        raw=raw, kind=classification.kind, confidence=classification.confidence,
        retrieval_query=retrieval or raw, as_of=as_of, compare_dates=compare,
        scope=scope, scope_evidence=dict(evidence), dates=dates, citations=citations,
        abbreviations=abbreviations_in(retrieval), needs_multi_hop=multi_hop,
        signals=signals, inherited=inherited)


def decontextualize(query: str, history: Sequence[QueryPlan], *,
                    today: date | None = None) -> QueryPlan:
    """Plan a follow-up turn against what the earlier turns established.

    The named entry point for the multi-turn case, and the one a chat loop should call. It is
    `plan_query` with history, spelled as the verb, because the thing that goes wrong here is
    forgetting to pass the history at all: the follow-up then plans cleanly, retrieves
    cleanly, and answers about the wrong person in the wrong year with full confidence.
    """
    return plan_query(query, history=history, today=today)
