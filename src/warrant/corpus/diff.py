"""Classifying what changed between two point-in-time snapshots of a part.

The temporal benchmark is mined from real amendments, so the value of the whole benchmark
depends on telling a real amendment apart from publication churn. Six classes, and only one
of them is usable as ground truth:

  substantive_localized   stable identifier, alignable text, localized semantic change  -> usable
  wholesale_rewrite       section replaced outright; no paragraph-level before/after
  editorial               punctuation, case or whitespace only
  apparatus_only          differs only in material stripped by corpus.apparatus
  renumbered              text preserved under a new identifier
  added / removed         section appears or disappears

The discard rate is reported, not hidden. A benchmark that silently drops 70% of its source
material is making a claim about representativeness that it has not earned.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from enum import StrEnum

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")

#: Text below this similarity is a replacement, not an amendment: before/after cannot be
#: aligned at paragraph level, so it cannot ground a "what changed" question.
WHOLESALE_THRESHOLD = 0.50
#: A change smaller than this is indistinguishable from typographic tidying.
MIN_CHANGED_TOKENS = 3
#: Text similarity above which a removed/added identifier pair is treated as renumbering.
RENUMBER_THRESHOLD = 0.90


class Change(StrEnum):
    SUBSTANTIVE = "substantive_localized"
    WHOLESALE = "wholesale_rewrite"
    EDITORIAL = "editorial"
    APPARATUS_ONLY = "apparatus_only"
    RENUMBERED = "renumbered"
    ADDED = "added"
    REMOVED = "removed"


@dataclass(frozen=True)
class SectionChange:
    identifier: str
    kind: Change
    from_date: str
    to_date: str
    similarity: float = 1.0
    changed_tokens: int = 0
    before: str = ""
    after: str = ""
    renamed_to: str | None = None

    @property
    def usable_for_benchmark(self) -> bool:
        return self.kind is Change.SUBSTANTIVE


def _loose(s: str) -> str:
    """If two texts match here but not exactly, the difference was punctuation/case only."""
    return _WS.sub(" ", _PUNCT.sub(" ", s.lower())).strip()


def _similarity(a: str, b: str) -> tuple[float, int]:
    at, bt = a.split(), b.split()
    sm = difflib.SequenceMatcher(None, at, bt, autojunk=False)
    changed = sum(max(i2 - i1, j2 - j1)
                  for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
    return sm.ratio(), changed


def classify_pair(before: str, after: str) -> tuple[Change, float, int]:
    if before == after:
        return Change.EDITORIAL, 1.0, 0
    if _loose(before) == _loose(after):
        return Change.EDITORIAL, 1.0, 0
    ratio, changed = _similarity(before, after)
    if changed < MIN_CHANGED_TOKENS:
        return Change.EDITORIAL, ratio, changed
    if ratio < WHOLESALE_THRESHOLD:
        return Change.WHOLESALE, ratio, changed
    return Change.SUBSTANTIVE, ratio, changed


def diff_snapshots(
    before: dict[str, str],
    after: dict[str, str],
    *,
    from_date: str,
    to_date: str,
    before_raw: dict[str, str] | None = None,
    after_raw: dict[str, str] | None = None,
) -> list[SectionChange]:
    """Compare two snapshots of one part.

    ``before``/``after`` map section identifier -> apparatus-stripped text. The optional
    ``*_raw`` maps carry the unstripped text; supplying them lets apparatus-only churn be
    counted rather than silently discarded, which is how the ingestion row of the failure
    budget stays honest.
    """
    changes: list[SectionChange] = []
    gone, fresh = set(before) - set(after), set(after) - set(before)

    renamed: dict[str, str] = {}
    for old in sorted(gone):
        for new in sorted(fresh):
            if new in renamed.values():
                continue
            ratio, _ = _similarity(before[old], after[new])
            if ratio >= RENUMBER_THRESHOLD:
                renamed[old] = new
                break

    for old, new in renamed.items():
        changes.append(SectionChange(old, Change.RENUMBERED, from_date, to_date,
                                     before=before[old], after=after[new], renamed_to=new))
    for ident in sorted(gone - set(renamed)):
        changes.append(SectionChange(ident, Change.REMOVED, from_date, to_date,
                                     before=before[ident]))
    for ident in sorted(fresh - set(renamed.values())):
        changes.append(SectionChange(ident, Change.ADDED, from_date, to_date,
                                     after=after[ident]))

    for ident in sorted(set(before) & set(after)):
        b, a = before[ident], after[ident]
        if b == a:
            if (before_raw is not None and after_raw is not None
                    and before_raw.get(ident) != after_raw.get(ident)):
                changes.append(SectionChange(ident, Change.APPARATUS_ONLY, from_date, to_date))
            continue
        kind, ratio, changed = classify_pair(b, a)
        changes.append(SectionChange(ident, kind, from_date, to_date,
                                     similarity=ratio, changed_tokens=changed,
                                     before=b, after=a))
    return changes
