"""Building the benchmark buckets.

Four buckets, **reported separately and never averaged into one number**. They measure
different things and have different difficulty, so a combined score would be a weighted
average of incomparable quantities whose weights nobody chose deliberately.

======================  =========================================  ====================
bucket                  ground truth from                          measures
======================  =========================================  ====================
``temporal``            real amendments between snapshots           dating correctness
``scope``               part-level applicability in the CFR titles  scope correctness
``scope-exclusion``     the same, inverted                          over-broad retrieval
``generated``           sampled in-force paragraphs                 retrieval coverage
``human``               hand-written, in ``benchmarks/human.yaml``   realistic queries
======================  =========================================  ====================

The temporal design is the one worth explaining. Each amendment yields **two** items that
share a query and differ only in ``as_of``: one dated inside the old validity interval, one
inside the new. The query is assembled only from wording the two versions share, so neither
version is favoured lexically and the as-of predicate is the only thing that can separate
them. A system without temporal filtering must get one of every pair wrong. A query carrying
wording unique to the in-force version would let plain lexical matching pick correctly by
accident, and the bucket would report a plausible number while testing nothing.

Known biases, stated here because the README quotes these buckets:

- ``temporal`` over-represents sections that change often; parts 890 and 315 dominate.
- ``generated`` questions are derived from the very paragraph they retrieve, so they are
  easier than real questions. The bucket measures coverage, not difficulty.
- ``human`` is author-written, not collected from users, and it is small. It characterizes
  the query distribution; it cannot rank configurations.
- Evidence sets start minimal and single, and grow by pooling as other configurations
  surface different sufficient evidence (ARCHITECTURE.md section 6).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import yaml

from ..corpus.diff import MIN_CHANGED_TOKENS, classify_pair
from ..index.store import Store
from ..retrieve.scope import GOVERNMENT_WIDE, PART_RESTRICTIONS, Scope

_WORD = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")
_STOP = frozenset("""
the and for that this with which shall must may not any all such other under upon into from
was were are been being has have had does did will would can could should than then when
where where who whom whose them they their there here its his her our your out off per via
each either neither both same only also more most less least very much many few some one two
section paragraph subpart part chapter title
""".split())
MAX_QUERY_TERMS = 14
#: Keep sampled dates clear of snapshot boundaries: eCFR dates a change to the snapshot in
#: which the text first differs, which need not be the day the amendment took effect.
BOUNDARY_MARGIN = timedelta(days=14)


@dataclass(frozen=True)
class BenchItem:
    id: str
    bucket: str
    query: str
    as_of: str
    section_id: str
    part: str
    heading: str
    #: Disjunction of minimal sufficient evidence sets. Any one set is enough. An empty
    #: disjunction member means nothing needs to be retrieved -- used by ``scope-exclusion``,
    #: where the whole question is whether something is *absent*.
    acceptable_evidence: list[list[str]]
    #: Retrieving one of these is the specific failure the bucket exists to detect.
    distractors: list[str] = field(default_factory=list)
    scope: Scope = GOVERNMENT_WIDE
    provenance: dict[str, str] = field(default_factory=dict)

    @property
    def all_evidence(self) -> list[str]:
        seen: dict[str, None] = {}
        for s in self.acceptable_evidence:
            for e in s:
                seen[e] = None
        return list(seen)

    def is_satisfied_by(self, retrieved: list[str]) -> bool:
        got = set(retrieved)
        return any(set(s) <= got for s in self.acceptable_evidence)

    def leaked(self, retrieved: list[str]) -> list[str]:
        d = set(self.distractors)
        return [c for c in retrieved if c in d]


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
    """A date comfortably inside a validity interval, or None if there is no room."""
    start = _iso(valid_from) + BOUNDARY_MARGIN
    end = (_iso(valid_to) if valid_to else _iso(horizon)) - BOUNDARY_MARGIN
    if start > end:
        return None
    return (start + (end - start) / 2).isoformat()


def salient_terms(text: str, *, limit: int = MAX_QUERY_TERMS) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for w in _WORD.findall(text):
        k = w.lower()
        if k in _STOP or k in seen:
            continue
        seen.add(k)
        out.append(w)
        if len(out) >= limit:
            break
    return out


def shared_query(before: str, after: str, heading: str) -> str:
    """A query from wording both versions share, so neither is favoured lexically."""
    b = {w.lower() for w in _WORD.findall(before)}
    a = {w.lower() for w in _WORD.findall(after)}
    shared = " ".join(w for w in _WORD.findall(before) if w.lower() in b & a)
    terms = salient_terms(shared)
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
                    e["paragraphs"]))
    for versions in out.values():
        versions.sort(key=lambda v: v.valid_from)
    return out


def _changed_anchors(old: Version, new: Version) -> list[str]:
    anchors = set(old.paragraphs) | set(new.paragraphs)
    return sorted(a for a in anchors
                  if old.paragraphs.get(a, ("", ""))[1] != new.paragraphs.get(a, ("", ""))[1])


# -- temporal ---------------------------------------------------------------------


def mine_temporal(store: Store, *, horizon: str,
                  min_changed_tokens: int = MIN_CHANGED_TOKENS) -> list[BenchItem]:
    """One item per amended paragraph per side.

    Per *paragraph*, not per section. An earlier version of this miner made a section's whole
    changed set the evidence for a single item, which produced 41 items requiring more
    paragraphs than the pipeline returns -- one needed 56 in a list of 8. Those were
    unsatisfiable by construction and were being reported as retrieval failures, quietly
    putting a floor under the bucket that no configuration could beat.

    Evidence sets must be *minimal*: what is needed to answer, not everything that happened to
    change on the same day. A paragraph and its counterpart in the neighbouring version are
    exactly that, and they are also the sharpest possible temporal contrast.
    """
    items: list[BenchItem] = []
    for section_id, versions in load_versions(store).items():
        for old, new in zip(versions, versions[1:], strict=False):
            _, _, section_changed = classify_pair(old.text, new.text)
            if section_changed < min_changed_tokens:
                continue
            for anchor in _changed_anchors(old, new):
                if anchor not in old.paragraphs or anchor not in new.paragraphs:
                    # A pure addition or deletion has no counterpart to confuse the retriever
                    # with, so it cannot test temporal discrimination.
                    continue
                old_id, old_text = old.paragraphs[anchor]
                new_id, new_text = new.paragraphs[anchor]
                kind, _, changed = classify_pair(old_text, new_text)
                if changed < min_changed_tokens:
                    continue
                query = shared_query(old_text, new_text, old.heading or new.heading)
                for label, version, evidence, distractor in (
                    ("before", old, old_id, new_id),
                    ("after", new, new_id, old_id),
                ):
                    as_of = sample_date(version.valid_from, version.valid_to,
                                        horizon=horizon)
                    if as_of is None:
                        continue
                    items.append(BenchItem(
                        id=f"{section_id}#{anchor}@{new.valid_from}:{label}",
                        bucket="temporal", query=query, as_of=as_of,
                        section_id=section_id, part=version.part, heading=version.heading,
                        acceptable_evidence=[[evidence]], distractors=[distractor],
                        provenance={"amended_on": new.valid_from, "side": label,
                                    "anchor": anchor, "change_kind": kind.value,
                                    "changed_tokens": str(changed),
                                    "valid_from": version.valid_from,
                                    "valid_to": version.valid_to or "open"}))
    return sorted(items, key=lambda i: i.id)


# -- scope ------------------------------------------------------------------------


def _current_versions(store: Store, horizon: str) -> list[Version]:
    out = []
    for versions in load_versions(store).values():
        live = [v for v in versions
                if v.valid_from <= horizon and (v.valid_to is None or v.valid_to > horizon)]
        out.extend(live)
    return sorted(out, key=lambda v: (v.part, v.section_id))


def mine_scope(store: Store, *, horizon: str, per_part: int = 12) -> list[BenchItem]:
    """Applicability items over the parts whose own titles restrict who they govern.

    ``scope`` items ask under a profile the part governs: the section must be retrieved, so
    the bucket catches a filter that is too aggressive. ``scope-exclusion`` items ask the
    same question under a profile the part does not govern: the section must be *absent*, so
    the bucket catches a filter that is too permissive. Reported separately, because a system
    can be perfect at one and useless at the other and a combined number would hide it.
    """
    items: list[BenchItem] = []
    by_part: dict[str, int] = {}
    for version in _current_versions(store, horizon):
        restriction = PART_RESTRICTIONS.get(version.part)
        if not restriction:
            continue
        facet, allowed = next(iter(restriction.items()))
        governed = sorted(allowed)[0]
        outside = _contrasting_value(facet, allowed)
        if outside is None:
            continue
        if by_part.get(version.part, 0) >= per_part:
            continue
        anchor = sorted(version.paragraphs)[0]
        version_id, text = version.paragraphs[anchor]
        terms = salient_terms(text)
        if len(terms) < 6:
            continue
        by_part[version.part] = by_part.get(version.part, 0) + 1
        query = f"{version.heading}: {' '.join(terms)}"
        as_of = horizon
        items.append(BenchItem(
            id=f"{version.section_id}:scope-in", bucket="scope", query=query, as_of=as_of,
            section_id=version.section_id, part=version.part, heading=version.heading,
            acceptable_evidence=[[version_id]], distractors=[],
            scope=Scope.of(**{facet: governed}),
            provenance={"facet": facet, "value": governed, "direction": "governs"}))
        items.append(BenchItem(
            id=f"{version.section_id}:scope-out", bucket="scope-exclusion", query=query,
            as_of=as_of, section_id=version.section_id, part=version.part,
            heading=version.heading,
            acceptable_evidence=[[]],   # nothing needs retrieving; absence is the answer
            distractors=[version_id],
            scope=Scope.of(**{facet: outside}),
            provenance={"facet": facet, "value": outside, "direction": "does not govern"}))
    return sorted(items, key=lambda i: i.id)


def _contrasting_value(facet: str, allowed: frozenset[str]) -> str | None:
    """A real value of the same facet that this part does not govern."""
    universe: set[str] = set()
    for restriction in PART_RESTRICTIONS.values():
        if facet in restriction:
            universe |= restriction[facet]
    outside = sorted(universe - allowed)
    return outside[0] if outside else None


# -- generated --------------------------------------------------------------------


def mine_generated(store: Store, *, horizon: str, stride: int = 9,
                   min_terms: int = 8) -> list[BenchItem]:
    """Coverage items over in-force paragraphs, sampled on a deterministic stride.

    A stride rather than a random sample so the bucket is reproducible without a seed. The
    query is built from the paragraph it retrieves, which makes these items easier than real
    questions -- the bucket measures whether the corpus is reachable at all, and its number
    should never be quoted as answer quality.
    """
    items: list[BenchItem] = []
    for i, version in enumerate(_current_versions(store, horizon)):
        if i % stride:
            continue
        anchor = sorted(version.paragraphs)[0]
        version_id, text = version.paragraphs[anchor]
        terms = salient_terms(text)
        if len(terms) < min_terms:
            continue
        items.append(BenchItem(
            id=f"{version.section_id}#{anchor}:gen", bucket="generated",
            query=f"{version.heading}: {' '.join(terms)}", as_of=horizon,
            section_id=version.section_id, part=version.part, heading=version.heading,
            acceptable_evidence=[[version_id]],
            provenance={"source": "paragraph-derived"}))
    return items


# -- human ------------------------------------------------------------------------


def load_human(path: Path, store: Store, *, horizon: str) -> list[BenchItem]:
    """Hand-written questions, with their evidence resolved against the live store.

    Evidence is written in the file as ``section#anchor`` and resolved here to the version in
    force on the item's date. Writing version ids by hand would rot the moment the corpus is
    rebuilt, and a benchmark whose ground truth silently stops resolving is worse than one
    that does not exist.
    """
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    items: list[BenchItem] = []
    for entry in raw:
        as_of = entry.get("as_of", horizon)
        sets: list[list[str]] = []
        for evidence_set in entry["evidence"]:
            resolved = [_resolve(store, ref, as_of) for ref in evidence_set]
            if all(resolved):
                sets.append([r for r in resolved if r])
        if not sets:
            raise ValueError(f"human item {entry['id']}: no evidence resolves at {as_of}")
        items.append(BenchItem(
            id=f"{entry['id']}:human", bucket="human", query=entry["query"], as_of=as_of,
            section_id=entry["evidence"][0][0].split("#")[0],
            part=entry.get("part", ""), heading=entry.get("heading", ""),
            acceptable_evidence=sets,
            scope=Scope.of(**entry.get("scope", {})),
            provenance={"source": "hand-written"}))
    return items


def _resolve(store: Store, ref: str, as_of: str) -> str | None:
    row = store.db.execute(
        "SELECT version_id FROM chunk WHERE chunk_id = ? AND system_to IS NULL "
        "AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)",
        (ref, as_of, as_of)).fetchone()
    return row["version_id"] if row else None


def mine_all(store: Store, *, horizon: str, human_path: Path | None = None
             ) -> dict[str, list[BenchItem]]:
    buckets: dict[str, list[BenchItem]] = {}
    for item in (mine_temporal(store, horizon=horizon)
                 + mine_scope(store, horizon=horizon)
                 + mine_generated(store, horizon=horizon)
                 + (load_human(human_path, store, horizon=horizon) if human_path else [])):
        buckets.setdefault(item.bucket, []).append(item)
    return buckets
