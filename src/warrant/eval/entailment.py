"""Scoring the entailment verifier against `benchmarks/entailment.yaml`.

`results/eval-007-entailment.md` reported 182 hand-labelled (claim, evidence) pairs, and the
labels lived only in a markdown appendix. Every number in that document was therefore a
transcription of a run nobody could repeat -- the same shape of claim this repository has
already been burned by, when `verify/entail.py` carried "71.3% on 148 hand-labelled pairs"
from a probe set that existed nowhere and survived review because nothing could check it.
This module makes the appendix executable.

Three properties are load-bearing, and each of them is a way the measurement could go quiet
rather than wrong:

**The strata are never pooled.** 129 pairs are what the generator actually emitted; 53 are
author-written minimal edits whose class balance was chosen, not observed. Pooled, the
adversarial contradictions repair a generator stratum that detects one contradiction in
three, and the headline stops saying anything about either distribution. `Report` has no
field holding a pooled accuracy, so pooling has to be written out by hand to happen at all.

**Micro and macro are both reported, per class.** The generator stratum is 86.8% micro and
60.1% macro because 102 of its 129 pairs are near-verbatim entailments. A verifier reported
on micro alone looks four points off its MNLI score while being at chance on the two classes
a grounding check exists to catch.

**An anchor that does not resolve is an error.** A pair whose premise has been amended out
of the store is not a pair to skip: skipping it shrinks the set while `n` in the report keeps
counting what the file says, and the benchmark quietly measures something smaller than it
claims to.

Intervals are section-clustered, from `warrant.eval.stats`. 182 pairs come from 91 sections
and one over-cited claim contributed ten of them; an item-level bootstrap treats those ten as
ten independent trials and reports an interval that is too narrow.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..config import REPO_ROOT
from ..index.store import Store
from ..verify import entail as E
from ..verify.align import align
from .stats import DEFAULT_SAMPLES, Interval, PairedDelta, cluster_bootstrap_ci, paired_delta

DEFAULT_PATH = REPO_ROOT / "benchmarks" / "entailment.yaml"

#: The two strata, in report order. Not a set: the order is the reading order of the results
#: doc, generator first because it is the observed distribution and the one a headline may
#: quote.
STRATA = ("generator", "adversarial")
GENERATOR, ADVERSARIAL = STRATA

LABELS = E.LABELS
_INDEX = {name: i for i, name in enumerate(LABELS)}

_REQUIRED = ("id", "stratum", "evidence", "as_of", "label", "claim")


class BenchmarkError(ValueError):
    """The benchmark file cannot be read as written."""


class UnresolvedEvidence(BenchmarkError):
    """A pair's `section#anchor` names no chunk the store believes on the pair's date."""


# -- the set ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Pair:
    """One labelled (premise, claim), with the premise resolved out of the store."""

    id: str
    stratum: str
    claim: str
    #: As written in the file: `section#anchor`, never a version id and never an offset.
    evidence: str
    as_of: str
    label: str
    #: Resolved at load time. Recorded so a report can name the exact text it scored.
    version_id: str
    premise: str
    section_id: str
    #: Which `human.yaml` question the generator was answering when it emitted this claim.
    #: Empty for the adversarial stratum, whose claims have no question behind them.
    question: str = ""

    @property
    def gold(self) -> int:
        return _INDEX[self.label]

    @property
    def gold_supported(self) -> bool:
        """The binary question span alignment can also answer: does the premise support it."""
        return self.label == "entail"


def load(path: str | Path = DEFAULT_PATH, *, store: Store) -> list[Pair]:
    """Read the benchmark and resolve every premise against the live store.

    Fails on the first pair that does not resolve rather than dropping it. The failure mode
    this guards is not a crash, it is silence: an amended paragraph makes one item stop being
    scored, the set shrinks, and the reported `n` -- which is read off the file -- does not
    move.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    pairs: list[Pair] = []
    seen: set[str] = set()
    for entry in raw:
        missing = [k for k in _REQUIRED if not str(entry.get(k, "")).strip()]
        if missing:
            raise BenchmarkError(f"{path.name}: pair {entry.get('id', '?')!r} is missing "
                                 f"{', '.join(missing)}")
        pair_id = str(entry["id"])
        if pair_id in seen:
            # Two pairs under one id double-count one measurement and make a per-pair
            # adjudication table ambiguous about which row it is talking about.
            raise BenchmarkError(f"{path.name}: duplicate pair id {pair_id!r}")
        seen.add(pair_id)
        if entry["stratum"] not in STRATA:
            raise BenchmarkError(
                f"{path.name}: pair {pair_id!r} has stratum {entry['stratum']!r}; the set is "
                f"reported per stratum and a third one would be averaged into neither")
        if entry["label"] not in LABELS:
            raise BenchmarkError(
                f"{path.name}: pair {pair_id!r} has label {entry['label']!r}, not one of "
                f"{'/'.join(LABELS)}")
        row = _resolve(store, str(entry["evidence"]), str(entry["as_of"]))
        if row is None:
            raise UnresolvedEvidence(
                f"{path.name}: pair {pair_id!r} cites {entry['evidence']} at "
                f"{entry['as_of']}, which the store does not believe. Re-anchor the pair or "
                f"remove it -- an unscored pair still counts toward the reported n.")
        pairs.append(Pair(
            id=pair_id, stratum=str(entry["stratum"]), claim=str(entry["claim"]),
            evidence=str(entry["evidence"]), as_of=str(entry["as_of"]),
            label=str(entry["label"]), version_id=row["version_id"], premise=row["text"],
            section_id=row["section_id"], question=str(entry.get("question", ""))))
    return pairs


def _resolve(store: Store, ref: str, as_of: str):
    return store.db.execute(
        "SELECT version_id, section_id, text FROM chunk WHERE chunk_id = ? "
        "AND system_to IS NULL AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)",
        (ref, as_of, as_of)).fetchone()


# -- results ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClassScore:
    label: str
    correct: int
    n: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0


@dataclass(frozen=True, slots=True)
class StratumScore:
    """One stratum's accuracy. There is deliberately no cross-stratum equivalent."""

    stratum: str
    n: int
    sections: int
    correct: int
    micro_ci: Interval
    per_class: tuple[ClassScore, ...]
    #: Rows gold, columns predicted, in `LABELS` order.
    confusion: tuple[tuple[int, int, int], ...]

    @property
    def micro(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def macro(self) -> float:
        """Unweighted mean over the classes the stratum actually contains.

        Classes with no pairs are left out rather than scored zero: the generator stratum
        supplies three contradictions and would otherwise be punished for a class balance
        that is the observation, not a choice.
        """
        present = [c for c in self.per_class if c.n]
        return sum(c.accuracy for c in present) / len(present) if present else 0.0

    def row(self) -> list[str]:
        cells = [self.stratum, str(self.n), str(self.sections),
                 f"{self.micro * 100:.1f}%", str(self.micro_ci), f"{self.macro * 100:.1f}%"]
        cells += [f"{c.correct}/{c.n}" if c.n else "-" for c in self.per_class]
        return cells


@dataclass(frozen=True, slots=True)
class SignalComparison:
    """Entailment against span alignment on one stratum, paired and section-clustered.

    Binary on both sides -- *does the premise support this claim* -- because that is the only
    question `align` can answer. `align` says supported when it locates a span; entailment
    says supported when the calibrated verdict reports `supported`.
    """

    stratum: str
    n: int
    agreement: float
    align_accuracy: float
    nli_accuracy: float
    delta: PairedDelta

    def row(self) -> list[str]:
        d = self.delta
        return [self.stratum, str(self.n), f"{self.agreement * 100:.1f}%",
                f"{self.align_accuracy * 100:.1f}%", f"{self.nli_accuracy * 100:.1f}%",
                f"{d.delta * 100:+.1f}", str(d.ci), f"{d.wins} / {d.losses}",
                f"{d.p_value:.2g}",
                "measurable" if d.significant else "not measurable"]


@dataclass(frozen=True, slots=True)
class PairResult:
    pair: Pair
    logits: tuple[float, float, float]
    probs: tuple[float, float, float]
    temperature: float
    predicted: str
    report: str
    span: bool

    @property
    def correct(self) -> bool:
        return self.predicted == self.pair.label


@dataclass(frozen=True, slots=True)
class Report:
    """Everything eval-007 section 1 and section 2 report, per stratum.

    No pooled accuracy field exists, on purpose. The pooled *confusion* does, because the
    results doc publishes one and counts add across strata without asserting anything about
    a mixture; an accuracy does not, and one written here would be quoted.
    """

    strata: tuple[StratumScore, ...]
    comparisons: tuple[SignalComparison, ...]
    pooled_confusion: tuple[tuple[int, int, int], ...]
    #: `None` when temperatures were fitted leave-one-section-out, which is what the results
    #: doc did; a float when a fixed temperature was passed in.
    fixed_temperature: float | None
    temperatures: dict[str, float] = field(default_factory=dict)
    results: list[PairResult] = field(default_factory=list)

    def stratum(self, name: str) -> StratumScore:
        for s in self.strata:
            if s.stratum == name:
                return s
        raise KeyError(name)

    def comparison(self, name: str) -> SignalComparison:
        for c in self.comparisons:
            if c.stratum == name:
                return c
        raise KeyError(name)

    def contradiction_rates(self) -> tuple[int, int, int]:
        """Pooled (true flags, missed, false flags) on the `contradict` argmax."""
        gold_c = _INDEX["contradict"]
        rows = self.pooled_confusion
        tp = rows[gold_c][gold_c]
        fn = sum(rows[gold_c]) - tp
        fp = sum(rows[g][gold_c] for g in range(3) if g != gold_c)
        return tp, fn, fp


# -- scoring ----------------------------------------------------------------------


def score(pairs: Sequence[Pair], *, entailer: E.Entailer | None = None,
          logits: Mapping[str, Sequence[float]] | None = None,
          temperature: float | None = None, samples: int = DEFAULT_SAMPLES,
          seed: int = 0) -> Report:
    """Score `entail.Entailer` and `align` over the set, per stratum and per class.

    `logits` replays a stored run instead of loading 377 MB of weights, which is what makes
    the scoring arithmetic -- the part most likely to be wrong in a way no metric would catch
    -- testable on a clone that has never downloaded a model.

    `temperature` defaults to a **leave-one-section-out** fit, so no pair's `supported`
    verdict is read off a temperature fitted on its own label. The fit pools both strata: it
    is one nuisance scalar over 182 pairs, and a per-stratum fit on 53 would be noisier than
    the quantity it corrects. Nothing else about the two strata is pooled. Pass the shipped
    `entail.CALIBRATION_TEMPERATURE` to score the way the serving path will.
    """
    pairs = list(pairs)
    raw = _logits(pairs, entailer=entailer, logits=logits)
    temps = ({p.id: temperature for p in pairs} if temperature is not None
             else _leave_one_section_out(pairs, raw))

    results: list[PairResult] = []
    for p in pairs:
        row = tuple(float(x) for x in raw[p.id])
        probs = E.softmax(row, temperature=temps[p.id])
        results.append(PairResult(
            pair=p, logits=row, probs=probs, temperature=temps[p.id],
            predicted=LABELS[max(range(3), key=lambda i: row[i])],
            report=E.Verdict(*probs).report,
            # `align` is asked the same question the verifier is: does this chunk support
            # this claim. A span is its yes.
            span=align(p.claim, p.premise) is not None))

    return Report(
        strata=tuple(_stratum_score(name, results, samples=samples, seed=seed)
                     for name in STRATA if any(r.pair.stratum == name for r in results)),
        comparisons=tuple(_comparison(name, results, samples=samples, seed=seed)
                          for name in STRATA if any(r.pair.stratum == name for r in results)),
        pooled_confusion=_confusion(results),
        fixed_temperature=temperature,
        temperatures=dict(temps),
        results=results,
    )


def _logits(pairs: Sequence[Pair], *, entailer: E.Entailer | None,
            logits: Mapping[str, Sequence[float]] | None) -> dict[str, Sequence[float]]:
    if logits is not None:
        missing = [p.id for p in pairs if p.id not in logits]
        if missing:
            raise BenchmarkError(f"no logits for {len(missing)} pairs, first {missing[0]!r}")
        return {p.id: logits[p.id] for p in pairs}
    scorer = entailer or E.Entailer()
    # Input order, one call: `Entailer` never sorts batches by length, so a verdict does not
    # move because a different claim was scored beside it.
    scored = scorer.logits([(p.premise, p.claim) for p in pairs])
    return {p.id: row for p, row in zip(pairs, scored, strict=True)}


def _leave_one_section_out(pairs: Sequence[Pair],
                           raw: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """A temperature per pair, fitted on every section except the pair's own."""
    out: dict[str, float] = {}
    for section in sorted({p.section_id for p in pairs}):
        held_in = [p for p in pairs if p.section_id != section]
        t = E.fit_temperature([raw[p.id] for p in held_in], [p.gold for p in held_in])
        for p in pairs:
            if p.section_id == section:
                out[p.id] = t
    return out


def _confusion(results: Sequence[PairResult]) -> tuple[tuple[int, int, int], ...]:
    matrix = [[0, 0, 0] for _ in LABELS]
    for r in results:
        matrix[r.pair.gold][_INDEX[r.predicted]] += 1
    return tuple(tuple(row) for row in matrix)


def _stratum_score(stratum: str, results: Sequence[PairResult], *, samples: int,
                   seed: int) -> StratumScore:
    sub = [r for r in results if r.pair.stratum == stratum]
    ok = [r.correct for r in sub]
    keys = [r.pair.section_id for r in sub]
    return StratumScore(
        stratum=stratum, n=len(sub), sections=len(set(keys)), correct=sum(ok),
        micro_ci=cluster_bootstrap_ci(ok, keys, samples=samples, seed=seed),
        per_class=tuple(
            ClassScore(label=name,
                       correct=sum(1 for r in sub if r.pair.label == name and r.correct),
                       n=sum(1 for r in sub if r.pair.label == name))
            for name in LABELS),
        confusion=_confusion(sub))


def _comparison(stratum: str, results: Sequence[PairResult], *, samples: int,
                seed: int) -> SignalComparison:
    sub = [r for r in results if r.pair.stratum == stratum]
    nli = [r.report == E.SUPPORTED for r in sub]
    span = [r.span for r in sub]
    gold = [r.pair.gold_supported for r in sub]
    nli_ok = [a == g for a, g in zip(nli, gold, strict=True)]
    align_ok = [a == g for a, g in zip(span, gold, strict=True)]
    n = len(sub) or 1
    return SignalComparison(
        stratum=stratum, n=len(sub),
        agreement=sum(1 for a, b in zip(nli, span, strict=True) if a == b) / n,
        align_accuracy=sum(align_ok) / n, nli_accuracy=sum(nli_ok) / n,
        delta=paired_delta(nli_ok, align_ok, [r.pair.section_id for r in sub],
                           samples=samples, seed=seed))
