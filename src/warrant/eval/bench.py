"""Mining the temporal benchmark out of real amendments.

Every item here is grounded in a change that actually happened to the law, so the ground
truth is not a judgement call: the text differed on two dates, and the eCFR version records
say when.

The design decision that makes the bucket discriminating is **how the query is built**. It is
assembled only from wording the two versions share. A query containing wording unique to the
in-force version would let plain lexical matching pick the right version by accident, and the
bucket would measure nothing. Built from shared wording, both versions of the section look
equally attractive to the retriever, and the only thing that can separate them is the as-of
predicate. A system without temporal filtering scores near zero on this bucket by
construction; a system with it should score near the ceiling.

Each item carries the losing version explicitly as a distractor. Retrieving it is not a near
miss, it is the specific failure the bucket exists to detect: quoting superseded law as
current.

Known biases, stated here because the README quotes this bucket:

- Questions are shaped by an amendment having occurred, so the bucket over-represents
  sections that change often. Parts 532 and 890 dominate.
- Queries are assembled from regulatory wording, not written by a person. They are more
  literal than real questions. That is what the human bucket is for, and the two are never
  averaged together.
- Evidence sets start minimal and single. They grow through pooling as other configurations
  surface different sufficient evidence (ARCHITECTURE.md section 6).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from ..corpus.diff import MIN_CHANGED_TOKENS, classify_pair
from ..index.store import Store

#: Wording shared by both versions, used to build a version-neutral query.
_WORD = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")
#: Function words carry no retrieval signal and make queries read like noise.
_STOP = frozenset("""
the and for that this with which shall must may not any all such other under upon into from
was were are been being has have had does did will would can could should than then when
where who whom whose them they their there here its his her our your out off per via each
either neither both same only also more most less least very much many few some one two
""".split())
#: A query longer than this stops being a query and starts being the document.
MAX_QUERY_TERMS = 14
#: Keep sampled dates clear of snapshot boundaries: eCFR dates a change to the snapshot in
#: which the text first differs, which is not necessarily the day the amendment took effect.
BOUNDARY_MARGIN = timedelta(days=14)


@dataclass(frozen=True)
class TemporalItem:
    id: str
    query: str
    as_of: str
    section_id: str
    part: str
    heading: str
    #: Disjunction of minimal sufficient evidence sets. Any one set is enough.
    acceptable_evidence: list[list[str]]
    #: The superseded or not-yet-in-force version of the same paragraphs. Retrieving one of
    #: these is the failure this bucket exists to detect.
    distractors: list[str]
    provenance: dict[str, str] = field(default_factory=dict)

    def is_satisfied_by(self, retrieved: list[str]) -> bool:
        got = set(retrieved)
        return any(set(s) <= got for s in self.acceptable_evidence)

    def leaked(self, retrieved: list[str]) -> list[str]:
        return [c for c in retrieved if c in set(self.distractors)]


@dataclass(frozen=True)
class Version:
    section_id: str
    part: str
    heading: str
    valid_from: str
    valid_to: str | None
    paragraphs: dict[str, tuple[str, str]]  # anchor -> (version_id, text)

    @property
    def text(self) -> str:
        return " ".join(t for _, t in self.paragraphs.values())


def _iso(d: str) -> date:
    return date.fromisoformat(d)


def sample_date(valid_from: str, valid_to: str | None, *, horizon: str) -> str | None:
    """A date comfortably inside a validity interval, or None if there is no room.

    Intervals shorter than twice the boundary margin are skipped rather than sampled at
    their edge. A question dated one day after an amendment is a question about snapshot
    bookkeeping, not about the law.
    """
    start = _iso(valid_from) + BOUNDARY_MARGIN
    end = (_iso(valid_to) if valid_to else _iso(horizon)) - BOUNDARY_MARGIN
    if start > end:
        return None
    return (start + (end - start) / 2).isoformat()


def shared_query(before: str, after: str, heading: str) -> str:
    """A query from wording both versions share, so neither is favoured lexically."""
    b = {w.lower() for w in _WORD.findall(before)}
    a = {w.lower() for w in _WORD.findall(after)}
    shared = [w for w in _WORD.findall(before)
              if w.lower() in b & a and w.lower() not in _STOP]
    seen: set[str] = set()
    terms: list[str] = []
    for w in shared:
        k = w.lower()
        if k not in seen:
            seen.add(k)
            terms.append(w)
        if len(terms) >= MAX_QUERY_TERMS:
            break
    return f"{heading}: {' '.join(terms)}" if terms else heading


def load_versions(store: Store) -> dict[str, list[Version]]:
    """Every believed chunk grouped into section versions, oldest first."""
    rows = store.db.execute(
        "SELECT section_id, part, heading, valid_from, valid_to, anchor, version_id, text "
        "FROM chunk WHERE system_to IS NULL ORDER BY section_id, valid_from, id"
    ).fetchall()
    grouped: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["section_id"], r["valid_from"])
        entry = grouped.setdefault(key, {
            "part": r["part"], "heading": r["heading"] or "",
            "valid_to": r["valid_to"], "paragraphs": {},
        })
        entry["paragraphs"][r["anchor"] or "full"] = (r["version_id"], r["text"])

    out: dict[str, list[Version]] = {}
    for (section_id, valid_from), e in grouped.items():
        out.setdefault(section_id, []).append(
            Version(section_id, e["part"], e["heading"], valid_from, e["valid_to"],
                    e["paragraphs"])
        )
    for versions in out.values():
        versions.sort(key=lambda v: v.valid_from)
    return out


def _changed_anchors(old: Version, new: Version) -> list[str]:
    anchors = set(old.paragraphs) | set(new.paragraphs)
    return sorted(
        a for a in anchors
        if old.paragraphs.get(a, ("", ""))[1] != new.paragraphs.get(a, ("", ""))[1]
    )


def mine(store: Store, *, horizon: str,
         min_changed_tokens: int = MIN_CHANGED_TOKENS) -> list[TemporalItem]:
    """Build one item per side of every substantive amendment in the store.

    An amendment yields two items: one dated inside the old version interval, one inside the
    new. Both use the same query. A system that ignores the as-of date must get one of them
    wrong, which is the property that makes this bucket worth reporting on its own.
    """
    items: list[TemporalItem] = []
    for section_id, versions in load_versions(store).items():
        for old, new in zip(versions, versions[1:], strict=False):
            kind, _, changed = classify_pair(old.text, new.text)
            if changed < min_changed_tokens:
                continue
            anchors = _changed_anchors(old, new)
            if not anchors:
                continue

            query = shared_query(old.text, new.text, old.heading or new.heading)
            before_ids = [old.paragraphs[a][0] for a in anchors if a in old.paragraphs]
            after_ids = [new.paragraphs[a][0] for a in anchors if a in new.paragraphs]
            if not before_ids or not after_ids:
                # A pure addition or deletion has no counterpart version to confuse the
                # retriever with, so it cannot test temporal discrimination.
                continue

            for label, version, evidence, distractors in (
                ("before", old, before_ids, after_ids),
                ("after", new, after_ids, before_ids),
            ):
                as_of = sample_date(version.valid_from, version.valid_to, horizon=horizon)
                if as_of is None:
                    continue
                items.append(TemporalItem(
                    id=f"{section_id}@{new.valid_from}:{label}",
                    query=query,
                    as_of=as_of,
                    section_id=section_id,
                    part=version.part,
                    heading=version.heading,
                    acceptable_evidence=[evidence],
                    distractors=distractors,
                    provenance={
                        "amended_on": new.valid_from,
                        "side": label,
                        "change_kind": kind.value,
                        "changed_tokens": str(changed),
                        "valid_from": version.valid_from,
                        "valid_to": version.valid_to or "open",
                    },
                ))
    items.sort(key=lambda i: i.id)
    return items
