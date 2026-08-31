"""Does closing a dangling reference produce a better *answer*?

``retrieve.multihop`` follows the citations a retrieved paragraph makes and admits the targets
the evidence set does not hold. It works: the share of evidence sets carrying an unsatisfied
reference falls 70.8% -> 32.2% at ``hop_budget: 8, hop_depth: 3``. It ships **off**, because
every number behind that is an intermediate metric and the one outcome metric that was
measured -- sufficiency -- moved -0.88 points and never once moved up (results/eval-013 §7).

The reason the decision could not be settled was stated in that report and in eval-006 §5:
no generated answers existed on disk, so nobody could ask whether the paragraph the hop
admits actually reaches the sentence a reader acts on. This module runs that experiment. Two
configurations, identical items, and four families of measure over the answers themselves:

**Hallucination and citation precision** -- ``eval.generation`` already defines both, and
they are imported rather than restated. A second definition of "grounded claim" living in a
second module is how two reports come to disagree about the same run.

**Unstated conditions** -- ``verify.qualifier`` over the chunks the answer cites. This is the
outcome metric the whole feature was deferred on: 25.7% of in-force chunks carry a qualifier,
and the failure that matters is the true sentence with the exception dropped. Reported with
its denominator, because the hop *adds* chunks and every added chunk can add conditions --
an arm that drops more conditions in absolute terms while stating a larger share of them is
not worse, and a bare count cannot tell the two apart.

**Dangling references remaining** -- ``verify.xref`` over the same cited set. eval-013
measured this over the whole ``final_k`` evidence set; measured over what the answer actually
cited, it is the share of the reader's own reading list that leads nowhere.

**The abstention quadrant** -- answered/abstained against evidence present/absent, because a
system that abstains everywhere has a perfect hallucination rate.

**Generation is the expensive part and is run once.** ~7 s an answer at 29.2-29.9 tok/s,
serialised, so two arms over a few hundred items is hours of GPU. So the model runs in
``generate_answers`` and writes to a SQLite cache; ``score_from_cache`` reads it back and
needs neither the model nor the retriever, rebuilding each context from the version ids the
cache recorded. Re-scoring for a metric nobody had thought of yet is then seconds.

Two arms with the same context produce the same prompt, and the generator is greedy at
temperature 0 -- so the second arm's answer is the first arm's answer, and the cache serves it
rather than paying for it twice. That is not a shortcut past the comparison: an item whose
evidence set the hop did not change carries no information about the hop, and saying so is the
point of a paired design. The share of items where the context *did* change is reported
beside every delta, because it is the real sample size.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..generate.answer import MAX_CONTEXT_CHUNKS, Answer, Claim, excerpts_for, ground
from ..index.store import Store
from ..retrieve.hybrid import Retriever
from ..verify.qualifier import chapeau_ids, qualifiers_of, unstated_conditions
from ..verify.xref import corpus_chunk_ids, dangling_references, nameable_ids
from .bench import BenchItem
from .generation import GenerationReport, GenerationResult, _score_one
from .stats import DEFAULT_SAMPLES, Interval, PairedDelta, paired_delta, wilson_ci

#: Where generated answers live between runs. Under ``data/`` beside ``traces.sqlite3``,
#: because it is the same kind of artifact: reproducible in principle, expensive in practice.
DEFAULT_CACHE = Path("data/hop-answers.sqlite3")

Excerpt = tuple[str, str, str]              # version id, heading, text


# -- what one arm did to one item --------------------------------------------------


@dataclass(frozen=True, slots=True)
class Outcome:
    """One item, one configuration, scored every way this study measures."""

    item_id: str
    bucket: str
    section_id: str
    arm: str
    context: tuple[str, ...]                # version ids handed to the generator, in order
    cited: tuple[str, ...]                  # the subset the answer actually cited
    #: True when this arm ran the model. False means an identical prompt had already been
    #: generated for the other arm and the greedy answer was reused.
    generated: bool
    result: GenerationResult
    #: Conditional qualifiers carried by the cited chunks, and how many of them the answer
    #: neither states nor gestures at. The denominator is not decoration: the hop admits more
    #: chunks, so it can raise both numbers at once.
    conditions: int
    unstated: int
    #: The same pair over the whole context, cited or not -- the stricter reading, and the
    #: only one an abstention can be scored on.
    conditions_context: int
    unstated_context: int
    #: References out of the cited set whose target the corpus holds and the set does not.
    dangling: int
    dangling_context: int
    answer_text: str

    @property
    def abstained(self) -> bool:
        return self.result.abstained

    @property
    def hallucinated(self) -> bool:
        return self.result.claims > self.result.grounded_claims

    @property
    def states_conditions(self) -> bool:
        """No condition in the cited text was dropped. Vacuously true of an abstention."""
        return self.unstated == 0

    @property
    def references_closed(self) -> bool:
        return self.dangling == 0


@dataclass(frozen=True, slots=True)
class ArmSummary:
    """Everything one arm did to one bucket, as rates with intervals."""

    arm: str
    bucket: str
    n: int
    generated: int
    report: GenerationReport
    conditions: int
    unstated: int
    answers_dropping_a_condition: int
    #: The same over the whole context rather than the cited subset. Its denominator is
    #: roughly ten times larger -- an answer cites one or two of sixteen excerpts -- which is
    #: most of the resolution this study has on the condition metric.
    conditions_context: int
    unstated_context: int
    dangling: int
    answers_with_a_dangling_reference: int
    answered: int

    @property
    def unstated_rate(self) -> float:
        """Share of the conditions in cited text that the answer did not carry."""
        return self.unstated / self.conditions if self.conditions else 0.0

    @property
    def unstated_rate_ci(self) -> Interval:
        return wilson_ci(self.unstated, self.conditions)

    @property
    def unstated_per_answer(self) -> float:
        return self.unstated / self.answered if self.answered else 0.0

    @property
    def unstated_context_rate(self) -> float:
        return self.unstated_context / self.conditions_context if self.conditions_context \
            else 0.0

    @property
    def unstated_context_rate_ci(self) -> Interval:
        return wilson_ci(self.unstated_context, self.conditions_context)

    @property
    def dangling_per_answer(self) -> float:
        return self.dangling / self.answered if self.answered else 0.0

    def rows(self) -> list[tuple[str, str, str]]:
        r = self.report
        return [
            ("hallucination rate", f"{r.hallucination_rate * 100:.1f}%",
             str(r.hallucination_ci)),
            ("citation precision", f"{r.citation_precision * 100:.1f}%",
             str(r.citation_precision_ci)),
            ("conditions in cited text", str(self.conditions), ""),
            ("of them, unstated", f"{self.unstated} ({self.unstated_rate * 100:.1f}%)",
             str(self.unstated_rate_ci)),
            ("answers dropping >=1 condition",
             f"{self.answers_dropping_a_condition}/{self.answered}", ""),
            ("conditions in context", str(self.conditions_context), ""),
            ("of them, unstated",
             f"{self.unstated_context} ({self.unstated_context_rate * 100:.1f}%)",
             str(self.unstated_context_rate_ci)),
            ("dangling refs in cited set", f"{self.dangling_per_answer:.2f}/answer", ""),
            ("answers with >=1 dangling ref",
             f"{self.answers_with_a_dangling_reference}/{self.answered}", ""),
            ("answered, evidence present", str(r.answered_with_evidence), "correct"),
            ("abstained, evidence absent", str(r.abstained_without_evidence), "correct"),
            ("abstained, evidence present", str(r.abstained_with_evidence),
             "wrong — it had the answer"),
            ("answered, evidence absent", str(r.answered_without_evidence),
             "wrong — answered anyway"),
        ]


# -- paired comparison --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MeanDelta:
    """A paired difference of means with a section-clustered interval.

    ``stats`` carries the boolean case and nothing else, because until now every paired
    quantity in this repository was a flag. Conditions dropped per answer is a count, and
    dichotomising it to "dropped any" throws away the difference between an answer that lost
    one exception and one that lost five. Same estimator as ``stats.paired_delta`` -- whole
    clusters resampled with replacement, pooled difference over pooled item count -- so the
    two are read the same way.
    """

    delta: float
    ci: Interval
    a_mean: float
    b_mean: float
    improved: int                   # items where A carries strictly fewer
    worsened: int
    p_value: float

    @property
    def significant(self) -> bool:
        return (self.ci.lo > 0 or self.ci.hi < 0) and self.p_value < 0.05


def _mcnemar_p(wins: int, losses: int) -> float:
    """Exact two-sided binomial test on the discordant pairs. Same test ``stats`` runs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_mean_delta(a: Sequence[float], b: Sequence[float], keys: Sequence[str], *,
                      samples: int = DEFAULT_SAMPLES, seed: int = 0) -> MeanDelta:
    """Mean of A minus mean of B on the same items, resampling sections."""
    groups: dict[str, list[tuple[float, float]]] = {}
    for x, y, key in zip(a, b, keys, strict=True):
        groups.setdefault(key, []).append((x, y))
    clusters = list(groups.values())
    n_items = len(a) or 1

    rng = random.Random(seed)
    n = len(clusters)
    deltas: list[float] = []
    for _ in range(samples):
        diff = 0.0
        total = 0
        for _ in range(n):
            picked = clusters[rng.randrange(n)]
            diff += sum(x - y for x, y in picked)
            total += len(picked)
        deltas.append(diff / total if total else 0.0)
    deltas.sort()
    ci = Interval(deltas[int(0.025 * (samples - 1))],
                  deltas[int(math.ceil(0.975 * (samples - 1)))])
    better = sum(1 for x, y in zip(a, b, strict=True) if x < y)
    worse = sum(1 for x, y in zip(a, b, strict=True) if x > y)
    return MeanDelta(delta=(sum(a) - sum(b)) / n_items, ci=ci,
                     a_mean=sum(a) / n_items, b_mean=sum(b) / n_items,
                     improved=better, worsened=worse, p_value=_mcnemar_p(better, worse))


@dataclass(frozen=True, slots=True)
class Comparison:
    """Two arms over one bucket, paired item by item.

    ``moved`` is the number of items whose context the hop actually changed. It is the real
    sample size of every row here: an item the hop left alone is graded against a byte-
    identical prompt and contributes a guaranteed tie, which narrows nothing and proves
    nothing.
    """

    bucket: str
    a: str                                  # the arm under test -- the hop
    b: str                                  # the reference -- the shipped configuration
    n: int
    moved: int
    flags: dict[str, PairedDelta] = field(default_factory=dict)
    means: dict[str, MeanDelta] = field(default_factory=dict)

    def rows(self) -> list[list[str]]:
        out: list[list[str]] = []
        for name, d in self.flags.items():
            out.append([name, f"{d.delta * 100:+.2f} pts", str(d.ci),
                        str(d.wins), str(d.losses), f"{d.p_value:.3g}"])
        for name, m in self.means.items():
            out.append([name, f"{m.delta:+.3f}", f"{m.ci.lo:.3f} to {m.ci.hi:.3f}",
                        str(m.improved), str(m.worsened), f"{m.p_value:.3g}"])
        return out


#: Item-level flags, in the direction where True is the good outcome, so that ``wins`` reads
#: as "items the hop got right and the shipped configuration did not" without a sign flip.
_FLAGS: dict[str, Callable[[Outcome], bool]] = {
    "evidence in context": lambda o: o.result.retrieved_evidence,
    "no hallucinated claim": lambda o: not o.hallucinated,
    "cited a gold chunk": lambda o: o.result.cited_gold,
    "stated every condition": lambda o: o.states_conditions,
    "stated every condition (context)": lambda o: o.unstated_context == 0,
    "no dangling reference": lambda o: o.references_closed,
    "answered": lambda o: not o.abstained,
}

_MEANS: dict[str, Callable[[Outcome], float]] = {
    "conditions dropped / answer": lambda o: float(o.unstated),
    "conditions in cited text": lambda o: float(o.conditions),
    "conditions dropped / answer (context)": lambda o: float(o.unstated_context),
    "conditions in context": lambda o: float(o.conditions_context),
    "dangling refs / answer": lambda o: float(o.dangling),
    "dangling refs / answer (context)": lambda o: float(o.dangling_context),
    "claims / answer": lambda o: float(o.result.claims),
}


# -- the corpus side of the metrics --------------------------------------------------


@dataclass(frozen=True, slots=True)
class _CorpusIndex:
    """The two membership sets the verifiers want, kept apart on purpose.

    Both are ``frozenset[str]`` and both are derived from the same chunk ids, and
    ``qualifiers_of`` and ``dangling_references`` each take whichever one they are handed
    without complaint -- ``in_corpus`` means "ids a chapeau could govern" in one and "every
    address the corpus can name" in the other. Passing the wrong one silently changes what is
    measured, so neither is ever a bare argument here.
    """

    nameable: frozenset[str]
    chapeaus: frozenset[str]


class CorpusIndexes:
    """``_CorpusIndex`` per ``as_of``, built once and reused.

    A temporal bucket asks about 61 distinct dates over 233 items, and each index is a full
    in-force scan plus two passes over ~10,000 chunk ids. Rebuilding it per item was 4x the
    cost of the scoring it feeds.
    """

    def __init__(self, store: Store) -> None:
        self.store = store
        self._cache: dict[str, _CorpusIndex] = {}

    def for_date(self, as_of: str) -> _CorpusIndex:
        got = self._cache.get(as_of)
        if got is None:
            ids = corpus_chunk_ids(self.store, as_of=as_of)
            got = _CorpusIndex(nameable=nameable_ids(ids), chapeaus=chapeau_ids(ids))
            self._cache[as_of] = got
        return got


# -- the answer cache ----------------------------------------------------------------


_SCHEMA = """
CREATE TABLE IF NOT EXISTS answer (
    arm           TEXT NOT NULL,
    item_id       TEXT NOT NULL,
    prompt_hash   TEXT NOT NULL,
    model         TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    -- The arm that actually paid for the generation. Differs from `arm` exactly where the
    -- two configurations built the same prompt, which is the population that carries no
    -- information about the hop and has to be countable afterwards.
    generated_by  TEXT NOT NULL,
    context       TEXT NOT NULL,          -- JSON list of version ids, in prompt order
    payload       TEXT NOT NULL,          -- JSON claims / answer_found / parse_failed / raw
    PRIMARY KEY (arm, item_id)
);
CREATE INDEX IF NOT EXISTS answer_prompt ON answer(prompt_hash);
"""


@dataclass(frozen=True, slots=True)
class Cached:
    arm: str
    item_id: str
    prompt_hash: str
    generated_by: str
    context: tuple[str, ...]
    claims: tuple[tuple[str, tuple[str, ...]], ...]     # (text, evidence version ids)
    answer_found: bool
    parse_failed: bool
    raw: str


def prompt_hash(model: str, question: str, excerpts: Sequence[Excerpt]) -> str:
    """Fingerprint of everything the generator is shown.

    The excerpt *texts* are hashed, not only their ids: a rebuilt corpus can re-address a
    paragraph or re-parse its text under an unchanged version id, and a cache that keyed on
    ids alone would serve an answer written from text that is no longer on disk.
    """
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(question.encode("utf-8"))
    for vid, heading, text in excerpts:
        for part in (vid, heading, text):
            h.update(b"\x00")
            h.update(part.encode("utf-8"))
    return h.hexdigest()[:32]


class AnswerCache:
    """Generated answers, keyed by (arm, item) and cross-indexed by prompt.

    Holds the model's output and the addresses it was shown, and nothing derived: every rate
    in this module is recomputed from it, so a metric added later is a re-score rather than a
    re-run. What it deliberately does not store is the chunk text -- that is in the store,
    keyed by the version ids recorded here, and duplicating it would put a second copy of the
    corpus on disk that could go stale against the first.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)

    def __enter__(self) -> AnswerCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    @staticmethod
    def _row(row: sqlite3.Row) -> Cached:
        payload = json.loads(row["payload"])
        return Cached(
            arm=row["arm"], item_id=row["item_id"], prompt_hash=row["prompt_hash"],
            generated_by=row["generated_by"], context=tuple(json.loads(row["context"])),
            claims=tuple((c["text"], tuple(c["evidence"])) for c in payload["claims"]),
            answer_found=bool(payload["answer_found"]),
            parse_failed=bool(payload["parse_failed"]), raw=payload.get("raw", ""))

    def get(self, arm: str, item_id: str) -> Cached | None:
        row = self.db.execute(
            "SELECT * FROM answer WHERE arm = ? AND item_id = ?", (arm, item_id)).fetchone()
        return self._row(row) if row else None

    def by_prompt(self, fingerprint: str) -> Cached | None:
        """Any arm's answer to a byte-identical prompt. Greedy decoding makes it this arm's."""
        row = self.db.execute(
            "SELECT * FROM answer WHERE prompt_hash = ? ORDER BY arm LIMIT 1",
            (fingerprint,)).fetchone()
        return self._row(row) if row else None

    def put(self, record: Cached, *, model: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO answer (arm, item_id, prompt_hash, model, created_at, "
            "generated_by, context, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (record.arm, record.item_id, record.prompt_hash, model,
             datetime.now(UTC).isoformat(timespec="seconds"), record.generated_by,
             json.dumps(list(record.context)),
             json.dumps({"claims": [{"text": t, "evidence": list(e)}
                                    for t, e in record.claims],
                         "answer_found": record.answer_found,
                         "parse_failed": record.parse_failed, "raw": record.raw})))
        self.db.commit()

    def arms(self) -> list[str]:
        return [r[0] for r in self.db.execute(
            "SELECT DISTINCT arm FROM answer ORDER BY arm")]


# -- generating -----------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One configuration under test, and the retriever that realises it."""

    name: str
    retriever: Retriever


def generate_answers(store: Store, items: Sequence[BenchItem], arms: Sequence[Arm],
                     generator, *, cache: AnswerCache,
                     context_k: int = MAX_CONTEXT_CHUNKS,
                     progress: Callable[[int, int, str], None] | None = None) -> dict[str, int]:
    """Retrieve and generate for every (item, arm), writing to ``cache``. Resumable.

    The only step here that needs a GPU. An item already in the cache under the same prompt
    is skipped, so an interrupted run resumes where it stopped, and an arm whose context
    matches an arm already generated for reuses the answer rather than decoding it again.
    """
    model = getattr(generator, "model_name", "unknown")
    counts = {"generated": 0, "reused": 0, "cached": 0}
    for i, item in enumerate(items, start=1):
        for arm in arms:
            if progress:
                progress(i, len(items), arm.name)
            got = cache.get(arm.name, item.id)
            trace = arm.retriever.retrieve(item.query, as_of=item.as_of, scope=item.scope)
            excerpts = excerpts_for(store, trace, limit=context_k)
            fingerprint = prompt_hash(model, item.query, excerpts)
            if got is not None and got.prompt_hash == fingerprint:
                counts["cached"] += 1
                continue
            twin = cache.by_prompt(fingerprint)
            if twin is not None:
                record = Cached(arm=arm.name, item_id=item.id, prompt_hash=fingerprint,
                                generated_by=twin.generated_by,
                                context=tuple(v for v, _, _ in excerpts),
                                claims=twin.claims, answer_found=twin.answer_found,
                                parse_failed=twin.parse_failed, raw=twin.raw)
                counts["reused"] += 1
            else:
                answer = generator.answer(item.query, excerpts, as_of=item.as_of,
                                          scope=item.scope.describe())
                record = Cached(
                    arm=arm.name, item_id=item.id, prompt_hash=fingerprint,
                    generated_by=arm.name, context=tuple(v for v, _, _ in excerpts),
                    claims=tuple((c.text, tuple(c.evidence)) for c in answer.claims),
                    answer_found=answer.answer_found, parse_failed=answer.parse_failed,
                    raw=answer.raw)
                counts["generated"] += 1
            cache.put(record, model=model)
    return counts


# -- scoring ---------------------------------------------------------------------------


def excerpts_from_ids(store: Store, version_ids: Sequence[str]) -> list[Excerpt]:
    """Rebuild a prompt's excerpts from the ids the cache recorded.

    This is what lets scoring run with no retriever and no model: the context is an ordered
    list of addresses, and the store is the thing that turns an address into text. Ids the
    store no longer holds are dropped, which shows up as a shorter context rather than as a
    crash -- and the prompt hash is what catches it if the text moved underneath.
    """
    if not version_ids:
        return []
    rows = {r["version_id"]: r for r in store.db.execute(
        f"SELECT version_id, heading, text FROM chunk WHERE version_id IN "
        f"({','.join('?' * len(version_ids))})", list(version_ids))}
    return [(v, rows[v]["heading"] or "", rows[v]["text"]) for v in version_ids if v in rows]


def _answer_of(item: BenchItem, record: Cached, excerpts: Sequence[Excerpt]) -> Answer:
    """The ``Answer`` the generator returned, rebuilt from the cache.

    ``ground`` is re-run rather than stored: the span aligner is deterministic and free, and
    a stored span would be an assertion about text this function is looking at anyway.
    """
    cited = {vid: text for vid, _heading, text in excerpts}
    claims = [Claim(text=t, evidence=list(e)) for t, e in record.claims]
    return Answer(question=item.query, as_of=item.as_of, scope=item.scope.describe(),
                  claims=ground(claims, cited), answer_found=record.answer_found,
                  cited=cited, raw=record.raw, parse_failed=record.parse_failed)


def _counts(answer_text: str, evidence: Mapping[str, str],
            index: _CorpusIndex) -> tuple[int, int, int]:
    """Conditions carried, conditions dropped, and references left dangling."""
    carried = sum(1 for qs in qualifiers_of(evidence, in_corpus=index.chapeaus).values()
                  for q in qs if q.conditional)
    dropped = len(unstated_conditions(answer_text, evidence, in_corpus=index.chapeaus))
    dangling = sum(1 for d in dangling_references(evidence, in_corpus=index.nameable)
                   if d.status == "missing")
    return carried, dropped, dangling


def score_outcome(item: BenchItem, arm: str, record: Cached, excerpts: Sequence[Excerpt],
                  index: _CorpusIndex) -> Outcome:
    answer = _answer_of(item, record, excerpts)
    context_ids = {vid for vid, _, _ in excerpts}
    result = _score_one(item, answer, context_ids,
                        retrieved_evidence=item.is_satisfied_by(list(context_ids)))
    used = [vid for c in answer.claims for vid in c.evidence]
    cited = {vid: answer.cited[vid] for vid in dict.fromkeys(used) if vid in answer.cited}
    text = answer.text()
    conditions, unstated, dangling = _counts(text, cited, index)
    ctx_conditions, ctx_unstated, ctx_dangling = _counts(text, answer.cited, index)
    return Outcome(
        item_id=item.id, bucket=item.bucket, section_id=item.section_id or item.id, arm=arm,
        context=tuple(vid for vid, _, _ in excerpts), cited=tuple(cited),
        generated=record.generated_by == arm, result=result,
        conditions=conditions, unstated=unstated,
        conditions_context=ctx_conditions, unstated_context=ctx_unstated,
        dangling=dangling, dangling_context=ctx_dangling, answer_text=text)


@dataclass(frozen=True)
class Study:
    """Every arm's outcomes over the same items, and the comparisons between them."""

    arms: tuple[str, ...]
    outcomes: dict[str, dict[str, Outcome]]     # arm -> item id -> outcome
    items: tuple[BenchItem, ...]

    def buckets(self) -> list[str]:
        return sorted({i.bucket for i in self.items})

    def _for(self, arm: str, bucket: str | None) -> list[Outcome]:
        return [o for item in self.items
                if (bucket is None or item.bucket == bucket)
                and (o := self.outcomes[arm].get(item.id)) is not None]

    def summary(self, arm: str, bucket: str | None = None) -> ArmSummary:
        outcomes = self._for(arm, bucket)
        report = GenerationReport(n=len(outcomes),
                                  results=[o.result for o in outcomes])
        answered = [o for o in outcomes if not o.abstained]
        return ArmSummary(
            arm=arm, bucket=bucket or "all", n=len(outcomes),
            generated=sum(1 for o in outcomes if o.generated), report=report,
            conditions=sum(o.conditions for o in outcomes),
            unstated=sum(o.unstated for o in outcomes),
            answers_dropping_a_condition=sum(1 for o in answered if o.unstated),
            conditions_context=sum(o.conditions_context for o in outcomes),
            unstated_context=sum(o.unstated_context for o in outcomes),
            dangling=sum(o.dangling for o in outcomes),
            answers_with_a_dangling_reference=sum(1 for o in answered if o.dangling),
            answered=len(answered))

    def compare(self, a: str, b: str, bucket: str | None = None, *,
                samples: int = DEFAULT_SAMPLES, seed: int = 0) -> Comparison:
        """``a`` minus ``b``, paired on items both arms answered, clustered by section."""
        pairs = [(x, y) for item in self.items
                 if (bucket is None or item.bucket == bucket)
                 and (x := self.outcomes[a].get(item.id)) is not None
                 and (y := self.outcomes[b].get(item.id)) is not None]
        keys = [x.section_id for x, _ in pairs]
        flags = {name: paired_delta([f(x) for x, _ in pairs], [f(y) for _, y in pairs],
                                    keys, samples=samples, seed=seed)
                 for name, f in _FLAGS.items()}
        means = {name: paired_mean_delta([f(x) for x, _ in pairs], [f(y) for _, y in pairs],
                                         keys, samples=samples, seed=seed)
                 for name, f in _MEANS.items()}
        return Comparison(bucket=bucket or "all", a=a, b=b, n=len(pairs),
                          moved=sum(1 for x, y in pairs if x.context != y.context),
                          flags=flags, means=means)


def score_from_cache(store: Store, items: Sequence[BenchItem], arms: Sequence[str], *,
                     cache: AnswerCache) -> Study:
    """Re-score whatever the cache holds. No model, no retriever, no GPU.

    Items with no cached answer for some arm are dropped from that arm rather than scored as
    abstentions -- a run that was interrupted has missing answers, and calling a missing
    answer an abstention would report the interruption as a result.
    """
    indexes = CorpusIndexes(store)
    outcomes: dict[str, dict[str, Outcome]] = {arm: {} for arm in arms}
    kept: list[BenchItem] = []
    for item in items:
        records = {arm: cache.get(arm, item.id) for arm in arms}
        if any(r is None for r in records.values()):
            continue
        kept.append(item)
        index = indexes.for_date(item.as_of)
        for arm, record in records.items():
            assert record is not None
            excerpts = excerpts_from_ids(store, record.context)
            outcomes[arm][item.id] = score_outcome(item, arm, record, excerpts, index)
    return Study(arms=tuple(arms), outcomes=outcomes, items=tuple(kept))
