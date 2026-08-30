"""Entailment between a claim and the paragraph it cites, reported as a calibrated signal.

`verify.align` answers *where in this chunk is the supporting text*. It answers it with
lexical overlap, so it says yes to a claim that reuses the paragraph's vocabulary and
reverses its meaning: "an employee **may** schedule restored leave" and "an employee
**shall** schedule restored leave" share every content word the aligner counts. Entailment is
the signal that can, in principle, tell those apart.

**It is a signal, not an authority.** Nothing here vetoes a claim. The module reports three
probabilities, a confidence, and whether that confidence lands in a band where the model has
been measured to be worth believing; the decision of what to do with an unsupported claim
belongs to the abstention policy, and the text of the regulation belongs to the reader. A
verifier that silently dropped claims would be substituting a 184M-parameter model trained on
newswire for 5 CFR, and it would do so with no record that it had.

The measured reason for that caution is in `docs/results/eval-007-entailment.md`, and it is
not the reason that was expected. Domain shift does not show up as a collapsed headline: on
129 pairs the generator actually emitted against 5 CFR, the model is right 86.8% of the time,
a few points under its published MNLI score. It shows up in the class breakdown. Those 129
pairs are 97% correct on entailment, **50% on neutral and 33% on contradiction** -- the
headline is carried entirely by the generator's habit of copying its premise nearly verbatim,
and on the two classes a verifier exists to catch, the model is at or near chance.

Against `verify.align` on the same 129 pairs the difference is **+2.3 points, p = 0.55, not
measurable** -- the same verdict this repository already reached about its cross-encoder
reranker, and it is reported here the same way. On 53 hand-written minimal edits of real
regulatory text -- one modality flipped, one number changed, one deadline moved -- it is
+49.1 points (p = 9e-7). So the *contradiction* channel is what this module buys, because it
is the one thing lexical overlap structurally cannot do; the entail/neutral boundary is not,
and is never used as a gate.

Premise is the regulation, hypothesis is the claim, and never the other way round. NLI is
directional: regulatory text routinely entails a claim it does not resemble, and a claim
almost never entails the paragraph it was drawn from.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field

#: Sentence boundaries come from the aligner rather than a second regex here. The two signals
#: are compared pair-for-pair in the results doc, and a comparison in which the two disagree
#: about where a sentence ends measures the split, not the models.
from .align import Span, _sentences

DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
#: Order is not assumed -- it is read from the checkpoint's ``id2label`` at load time. This
#: model's head runs entailment/neutral/contradiction; `cross-encoder/nli-deberta-v3-base`,
#: the other candidate, runs contradiction/entailment/neutral. A hard-coded index 0 reads the
#: second checkpoint's contradictions as support, with every published number still in range.
LABELS = ("entail", "neutral", "contradict")
_ALIASES = {"entailment": "entail", "entail": "entail",
            "neutral": "neutral",
            "contradiction": "contradict", "contradict": "contradict"}

#: DeBERTa-v3's positional embeddings stop here. Measured on the 9,961 in-force chunks: p99
#: is 209 tokens and exactly one chunk exceeds 500, so the windowing path below is all but
#: dead code. It exists because that one is a long procedural section, which is exactly the
#: shape of text where a truncated premise drops the proviso that decides the answer.
MAX_LENGTH = 512
#: Chosen by measurement, not by feel: on an RTX 5070 Laptop throughput *peaks* at 16 (458
#: pairs/s, against 323 at 8 and 395 at 32) and weights plus batch-16 activations are 918 MB,
#: which co-exists with the 1.5B generator inside 8 GB.
DEFAULT_BATCH = 16

#: Temperature fitted by NLL on the 182-pair probe set (see the results doc); refits leaving
#: out one section at a time stay inside 1.66-1.74, so it is not one section's artefact. It
#: takes ECE from 9.5% to 4.0%. >1 means the raw head is overconfident, which is the expected
#: direction under domain shift: the model is as sure about regulatory prose as it was about
#: the captions it was trained on.
CALIBRATION_TEMPERATURE = 1.72

#: Above this calibrated confidence the model was right on 89.4% of 170 probe pairs; below
#: it, on 7 of 12. That gap is the whole justification for the band, and a verdict inside it
#: is reported as `uncertain` rather than believed in either direction. Raising the floor
#: further buys almost nothing -- at 0.85 accuracy above the line moves to 91.8% and coverage
#: falls to 74% -- so the band is set where the separation is, not where the accuracy is
#: highest.
DECISION_FLOOR = 0.70
#: Contradiction is reported at a *lower* bar than support. Asymmetric because the costs are:
#: a missed contradiction ships a claim that the regulation denies, a false one adds a flag a
#: human reads. The sweep in the results doc is flat across 0.40-0.60 (recall 76%, precision
#: 73%) and this is its middle; 0.30 scores two more true flags on 25 contradictions, which is
#: inside the noise of a set that small and not worth tuning to.
CONTRADICT_FLOOR = 0.50

SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
CONTRADICTED = "contradicted"
UNCERTAIN = "uncertain"


# -- verdicts ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Verdict:
    """One (premise, hypothesis) pair scored. Probabilities are calibrated, not raw."""

    entail: float
    neutral: float
    contradict: float
    #: Sentence window of the premise the scores came from, when the premise was too long
    #: for one pass. None means the whole chunk was scored, which is the usual case.
    window: Span | None = None

    @property
    def probs(self) -> tuple[float, float, float]:
        return (self.entail, self.neutral, self.contradict)

    @property
    def label(self) -> str:
        """The argmax, whether or not it is believable. Use ``report`` to get the band."""
        return LABELS[max(range(3), key=lambda i: self.probs[i])]

    @property
    def confidence(self) -> float:
        return max(self.probs)

    @property
    def report(self) -> str:
        """What this pair licenses saying: one of the four module-level constants.

        The argmax alone is not it. A 0.41/0.39/0.20 split has an argmax and says nothing,
        and publishing that as `entail` is how a signal becomes an unearned authority.
        """
        if self.contradict >= CONTRADICT_FLOOR:
            return CONTRADICTED
        if self.confidence < DECISION_FLOOR:
            return UNCERTAIN
        return SUPPORTED if self.label == "entail" else UNSUPPORTED


@dataclass(frozen=True, slots=True)
class ClaimSupport:
    """One claim against every chunk it cited.

    Aggregated by *any*, not by mean: a claim citing four paragraphs needs one of them to
    support it, and averaging in the three that merely sit nearby in the same section
    dilutes the only score that carried information. Contradiction is aggregated the same
    way and reported alongside rather than instead -- a claim can be entailed by the version
    in force and contradicted by the one it superseded, which for a temporal question is the
    correct answer rather than a conflict to resolve.
    """

    claim: str
    verdicts: dict[str, Verdict] = field(default_factory=dict)

    @property
    def best(self) -> tuple[str, Verdict] | None:
        if not self.verdicts:
            return None
        return max(self.verdicts.items(), key=lambda kv: kv[1].entail)

    @property
    def contradicted_by(self) -> list[str]:
        return sorted(cid for cid, v in self.verdicts.items()
                      if v.report == CONTRADICTED)

    @property
    def report(self) -> str:
        if not self.verdicts:
            return UNCERTAIN
        reports = {cid: v.report for cid, v in self.verdicts.items()}
        if SUPPORTED in reports.values():
            return SUPPORTED
        if CONTRADICTED in reports.values():
            return CONTRADICTED
        if UNCERTAIN in reports.values():
            return UNCERTAIN
        return UNSUPPORTED


def combine(span: Span | None, support: ClaimSupport) -> str:
    """The two grounding signals, reported together rather than one overruling the other.

    Span alignment is precise about *location* and blind to *polarity*; entailment is the
    reverse. So the combination that matters is not a vote -- it is the disagreement:

    - span and entailment agree -> that agreement is the thing worth reporting
    - a span exists and the model contradicts -> the citation points at text that denies the
      claim, which is the failure lexical overlap cannot see and the reason this module exists
    - no span and the model entails -> the claim is supported by wording the aligner could
      not match, so the citation is real and the *span* is what is missing
    - neither -> ungrounded, as `align` already said, with a second signal agreeing

    Returns one of the four module constants. It never returns "verified": no arrangement of
    these two signals establishes that a sentence about someone's leave entitlement is right.
    """
    nli = support.report
    if nli == CONTRADICTED:
        return CONTRADICTED
    if span is None:
        return nli if nli == SUPPORTED else UNSUPPORTED
    return SUPPORTED if nli == SUPPORTED else UNCERTAIN


# -- calibration ------------------------------------------------------------------
#
# Kept free of torch so the calibration path is unit-testable on a clone with no weights
# downloaded, which is most clones.


def softmax(logits: Sequence[float], *, temperature: float = 1.0) -> tuple[float, ...]:
    scaled = [x / temperature for x in logits]
    top = max(scaled)
    exps = [math.exp(x - top) for x in scaled]
    total = sum(exps)
    return tuple(e / total for e in exps)


def fit_temperature(logits: Sequence[Sequence[float]], gold: Sequence[int], *,
                    lo: float = 0.25, hi: float = 8.0, iterations: int = 60) -> float:
    """The single scalar that minimises NLL on a labelled set.

    One parameter, fitted on held-out labels, and nothing else: temperature scaling cannot
    change any argmax, so it cannot move accuracy. That is the point. It moves only the
    confidence attached to a decision the model was already making, which is the quantity a
    downstream abstention policy reads and the one the raw head gets wrong under domain
    shift. Anything richer -- per-class vectors, isotonic regression -- has more parameters
    than this probe set has pairs per class.

    Ternary search over a strictly unimodal objective; a gradient loop would need torch here
    for one scalar.
    """
    if not logits:
        return 1.0

    def nll(t: float) -> float:
        return -sum(math.log(max(softmax(row, temperature=t)[g], 1e-12))
                    for row, g in zip(logits, gold, strict=True)) / len(gold)

    for _ in range(iterations):
        a = lo + (hi - lo) / 3
        b = hi - (hi - lo) / 3
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a
    return (lo + hi) / 2


@dataclass(frozen=True, slots=True)
class Bin:
    lo: float
    hi: float
    n: int
    confidence: float      # mean predicted confidence in the bin
    accuracy: float        # share correct in the bin

    @property
    def gap(self) -> float:
        return self.confidence - self.accuracy


def reliability(confidence: Sequence[float], correct: Sequence[bool], *,
                bins: int = 10) -> list[Bin]:
    """Equal-width confidence bins, empty ones dropped.

    Equal-width rather than equal-mass because the question is *at 0.9, is it right 90% of
    the time* -- a fixed-count binning answers a differently-shaped question and hides the
    top bin, which is where an overconfident model does its damage.
    """
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for c, ok in zip(confidence, correct, strict=True):
        idx = min(bins - 1, max(0, int(c * bins)))
        buckets[idx].append((c, ok))
    out = []
    for i, rows in enumerate(buckets):
        if not rows:
            continue
        out.append(Bin(lo=i / bins, hi=(i + 1) / bins, n=len(rows),
                       confidence=sum(c for c, _ in rows) / len(rows),
                       accuracy=sum(1 for _, ok in rows if ok) / len(rows)))
    return out


def expected_calibration_error(confidence: Sequence[float], correct: Sequence[bool], *,
                               bins: int = 10) -> float:
    """Sample-weighted mean gap between confidence and accuracy."""
    n = len(confidence)
    if not n:
        return 0.0
    return sum(b.n * abs(b.gap) for b in reliability(confidence, correct, bins=bins)) / n


def brier(probs: Sequence[Sequence[float]], gold: Sequence[int]) -> float:
    """Multi-class Brier score. Reported beside ECE because ECE is blind to a model that is
    uniformly 60% confident and 60% accurate on every pair -- perfectly calibrated and
    useless."""
    if not probs:
        return 0.0
    total = 0.0
    for row, g in zip(probs, gold, strict=True):
        total += sum((p - (1.0 if i == g else 0.0)) ** 2 for i, p in enumerate(row))
    return total / len(probs)


# -- the model --------------------------------------------------------------------

#: One model per (name, revision) per process. Constructing it costs ~1.4 s from a warm disk
#: cache and 377 MB of VRAM, and a per-call construction would put the load time inside every
#: latency number this module publishes. Guarded because two cold threads both taking the
#: check-then-set branch is an out-of-memory error on an 8 GB card already holding the
#: generator.
_MODELS: dict[tuple[str, str | None, str], tuple] = {}
_MODEL_LOCK = threading.Lock()


def _load(model_name: str, revision: str | None, device: str,
          deterministic: bool) -> tuple:
    key = (model_name, revision, device)
    with _MODEL_LOCK:
        if key in _MODELS:
            return _MODELS[key]

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if deterministic:
            torch.manual_seed(0)
            # warn_only: this flag is process-global, and the co-resident 1.5B generator
            # uses kernels with no deterministic implementation. Raising there would make
            # importing the verifier break generation, which is a worse failure than a
            # warning. Nothing in DeBERTa inference is on the nondeterministic list, so the
            # verifier gets the guarantee and the generator keeps running.
            torch.use_deterministic_algorithms(True, warn_only=True)

        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, revision=revision,
            dtype=torch.float16 if device.startswith("cuda") else torch.float32)
        model.to(device)
        model.eval()

        order = _label_order(model.config.id2label)
        _MODELS[key] = (tokenizer, model, order)
        return _MODELS[key]


def _label_order(id2label: dict) -> tuple[int, int, int]:
    """Head indices for (entail, neutral, contradict), read off the checkpoint.

    Raises rather than guessing. A checkpoint whose labels are ``LABEL_0/1/2`` carries no
    statement of its own ordering, and assuming one produces a verifier that reports every
    contradiction as support -- scores stay in range, the pipeline stays green, and every
    published number is inverted.
    """
    found: dict[str, int] = {}
    for idx, name in id2label.items():
        alias = _ALIASES.get(str(name).strip().lower())
        if alias:
            found[alias] = int(idx)
    if set(found) != set(LABELS):
        raise ValueError(
            f"cannot read NLI label order from id2label={id2label!r}; expected names among "
            f"{sorted(_ALIASES)}")
    return (found["entail"], found["neutral"], found["contradict"])


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Entailer:
    """Scores (premise, hypothesis) pairs in batches, on whatever device is available.

    Stateless between calls apart from ``stats``, which accumulates the throughput this
    module is required to publish. A verifier whose cost is unmeasured is a verifier nobody
    can decide to put on the synchronous path.

    Measured, so that decision can be made: a whole answer is 2.4 (claim, chunk) pairs on
    average, one batched call, **24 ms p50 on GPU and ~200 ms on CPU**. Generation of the
    same answer takes about 20 s at 21.3 tok/s, so this is 0.1% of it on GPU and 1% on CPU.
    It belongs on the synchronous path; nothing about its cost justifies an async audit.
    """

    model_name: str = DEFAULT_MODEL
    revision: str | None = None
    device: str | None = None
    batch_size: int = DEFAULT_BATCH
    max_length: int = MAX_LENGTH
    #: 1.0 gives the raw head. The default is the fitted value; a caller re-fitting on its
    #: own labelled set passes its own.
    temperature: float = CALIBRATION_TEMPERATURE
    deterministic: bool = True
    stats: dict[str, float] = field(
        default_factory=lambda: {"pairs": 0, "batches": 0, "seconds": 0.0, "windowed": 0})

    @property
    def throughput(self) -> float:
        """Pairs per second over everything this instance has scored."""
        return self.stats["pairs"] / self.stats["seconds"] if self.stats["seconds"] else 0.0

    def logits(self, pairs: Sequence[tuple[str, str]]) -> list[tuple[float, float, float]]:
        """Raw head outputs, reordered to (entail, neutral, contradict).

        Batches are formed in input order and never sorted by length. Sorting is the obvious
        throughput win and it makes the result depend on which other pairs happened to be in
        the request: padding changes the float arithmetic, and a verdict that moves because a
        different claim was scored alongside it is not reproducible from a stored trace. The
        measured cost on this corpus is 455 against 560 pairs/s -- 19%, on a stage that is
        0.1% of answer latency.

        This buys reproducibility for a *fixed* batch size, not independence from batching.
        Re-scoring the probe set at batch 7 instead of 16 moved logits by up to 0.014 and no
        argmax at all, so `batch_size` belongs in a trace beside the model name.
        """
        import torch

        if not pairs:
            return []
        device = _resolve_device(self.device)
        tokenizer, model, order = _load(self.model_name, self.revision, device,
                                        self.deterministic)
        out: list[tuple[float, float, float]] = []
        started = time.perf_counter()
        for batch in _chunks(list(pairs), self.batch_size):
            encoded = tokenizer([p for p, _ in batch], [h for _, h in batch],
                                padding=True, truncation="only_first",
                                max_length=self.max_length, return_tensors="pt").to(device)
            with torch.no_grad():
                raw = model(**encoded).logits.float().cpu()
            out.extend((float(row[order[0]]), float(row[order[1]]), float(row[order[2]]))
                       for row in raw)
            self.stats["batches"] += 1
        self.stats["pairs"] += len(pairs)
        self.stats["seconds"] += time.perf_counter() - started
        return out

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[Verdict]:
        """Calibrated verdicts for (premise, hypothesis) pairs, in input order."""
        return [Verdict(*softmax(row, temperature=self.temperature))
                for row in self.logits(pairs)]

    def score_claim(self, claim: str, sources: dict[str, str]) -> ClaimSupport:
        """One claim against every chunk it cites, one batch for the whole claim.

        Premises longer than the encoder are split into sentence windows here rather than
        truncated. Truncation on a regulatory premise is not a rounding error: "Except as
        provided in paragraph (d)" and the paragraph (d) it excepts are routinely 600 tokens
        apart, and dropping the tail silently converts a conditional rule into an absolute
        one.
        """
        if not sources:
            return ClaimSupport(claim=claim)
        expanded: list[tuple[str, str]] = []
        origin: list[tuple[str, Span | None]] = []
        for chunk_id, text in sources.items():
            windows = self._windows(text, claim)
            self.stats["windowed"] += int(len(windows) > 1)
            for span in windows:
                expanded.append((text[span.start:span.end] if span else text, claim))
                origin.append((chunk_id, span))

        verdicts: dict[str, Verdict] = {}
        for (chunk_id, span), verdict in zip(origin, self.score(expanded), strict=True):
            scored = Verdict(*verdict.probs, window=span)
            current = verdicts.get(chunk_id)
            # The most *decisive* window wins, not the most entailing one. Taking the
            # entailment-max would let a chunk whose first sentence states the rule and whose
            # second revokes it report as clean support, which is the failure mode windowing
            # was introduced to avoid.
            if current is None or _decisiveness(scored) > _decisiveness(current):
                verdicts[chunk_id] = scored
        return ClaimSupport(claim=claim, verdicts=verdicts)

    def score_answer(self, claims: Iterable, cited: dict[str, str]) -> list[ClaimSupport]:
        """Every claim of an answer, against the chunks each one cites.

        Takes the duck-typed ``Claim`` of `generate.answer` -- anything with ``.text`` and
        ``.evidence`` -- rather than importing it, so the verifier does not depend on the
        generator it audits.
        """
        return [self.score_claim(c.text, {vid: cited.get(vid, "") for vid in c.evidence
                                          if cited.get(vid)})
                for c in claims]

    def _windows(self, premise: str, hypothesis: str) -> list[Span | None]:
        """``[None]`` when the pair fits whole, otherwise the sentence windows to score."""
        tokenizer, _model, _order = _load(self.model_name, self.revision,
                                          _resolve_device(self.device), self.deterministic)
        budget = self.max_length - len(tokenizer(hypothesis)["input_ids"]) - 4
        if budget <= 0 or len(tokenizer(premise)["input_ids"]) <= budget:
            return [None]

        spans: list[Span] = []
        sentences = _sentences(premise)
        start = 0
        while start < len(sentences):
            end = start
            used = 0
            while end < len(sentences):
                cost = len(tokenizer(premise[sentences[end][0]:sentences[end][1]])
                           ["input_ids"])
                if used and used + cost > budget:
                    break
                used += cost
                end += 1
            spans.append(Span(sentences[start][0], sentences[end - 1][1], 0.0))
            # One sentence of overlap: a rule and its exception are usually adjacent, and a
            # hard split between them scores both halves as something neither one says.
            start = end - 1 if end - 1 > start else end
        return spans or [None]


def _decisiveness(v: Verdict) -> float:
    return max(v.entail, v.contradict)


def _chunks(items: list, size: int) -> Iterator[list]:
    # Clamped once and used for both the step and the slice. Clamping only the step -- which
    # is the natural way to write this -- makes ``batch_size=0`` yield one empty list per
    # item, so every pair comes back unscored and the strict zip in `score_claim` raises
    # somewhere unrelated to the config that caused it.
    width = max(1, size)
    for i in range(0, len(items), width):
        yield items[i:i + width]
