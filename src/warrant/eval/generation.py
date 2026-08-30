"""Measuring the generator, not just the retriever.

The stage most likely to produce a wrong answer in a legal system is the model that writes
the sentence a reader will act on, and for a while it was the one stage nothing here scored.
The failure budget can now *attribute* a failure to generation or grounding; this module
*measures* it, which is a different question — attribution says which stage lost the
evidence, these numbers say how the model behaves when it had the evidence all along.

Three quantities, all computable from the existing ``Answer`` object with no extra model and
no new labelling, because the ground truth is the evidence set the benchmark already carries:

**Citation precision** — of the paragraphs the answer cited, what share were actually in the
context and carry a locatable supporting span. A citation to something the model was never
shown is a fabricated reference; a citation whose text does not support the claim is an
unsupported one. Both are wrong, and they are wrong differently, so both are counted.

**Hallucination rate** — the share of emitted claims with no locatable support in any chunk
they cite. This is the number a reader of a regulatory answer actually cares about.

**Abstention quality** — abstaining is correct when the evidence was not retrieved and wrong
when it was. Scored as a two-by-two rather than a single rate, because a system that always
abstains has a perfect hallucination rate and is useless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..generate.answer import Answer, excerpts_for
from ..retrieve.hybrid import Retriever
from .bench import BenchItem
from .stats import Interval, wilson_ci


@dataclass(frozen=True)
class GenerationResult:
    item_id: str
    retrieved_evidence: bool          # did retrieval put sufficient evidence in context
    abstained: bool
    parse_failed: bool
    claims: int
    grounded_claims: int
    citations: int
    good_citations: int               # in context AND with a locatable span
    cited_gold: bool                  # cited at least one sufficient-evidence chunk


@dataclass
class GenerationReport:
    n: int = 0
    results: list[GenerationResult] = field(default_factory=list)

    # -- abstention, as a two-by-two ---------------------------------------------

    @property
    def abstained_with_evidence(self) -> int:
        """Wrong: the answer was in the context and the model declined to give it."""
        return sum(1 for r in self.results if r.abstained and r.retrieved_evidence)

    @property
    def abstained_without_evidence(self) -> int:
        """Right: nothing sufficient was retrieved, and the model said so."""
        return sum(1 for r in self.results if r.abstained and not r.retrieved_evidence)

    @property
    def answered_with_evidence(self) -> int:
        return sum(1 for r in self.results if not r.abstained and r.retrieved_evidence)

    @property
    def answered_without_evidence(self) -> int:
        """Wrong, and the dangerous one: an answer written from insufficient context."""
        return sum(1 for r in self.results if not r.abstained and not r.retrieved_evidence)

    # -- rates --------------------------------------------------------------------

    @property
    def claims(self) -> int:
        return sum(r.claims for r in self.results)

    @property
    def hallucinated(self) -> int:
        return sum(r.claims - r.grounded_claims for r in self.results)

    @property
    def hallucination_rate(self) -> float:
        return self.hallucinated / self.claims if self.claims else 0.0

    @property
    def hallucination_ci(self) -> Interval:
        return wilson_ci(self.hallucinated, self.claims)

    @property
    def citations(self) -> int:
        return sum(r.citations for r in self.results)

    @property
    def citation_precision(self) -> float:
        good = sum(r.good_citations for r in self.results)
        return good / self.citations if self.citations else 0.0

    @property
    def citation_precision_ci(self) -> Interval:
        good = sum(r.good_citations for r in self.results)
        return wilson_ci(good, self.citations)

    @property
    def parse_failures(self) -> int:
        return sum(1 for r in self.results if r.parse_failed)

    def rows(self) -> list[tuple[str, str, str]]:
        answered = self.n - self.abstained_with_evidence - self.abstained_without_evidence
        return [
            ("claims emitted", str(self.claims), ""),
            ("hallucination rate", f"{self.hallucination_rate * 100:.1f}%",
             str(self.hallucination_ci)),
            ("citation precision", f"{self.citation_precision * 100:.1f}%",
             str(self.citation_precision_ci)),
            ("answered, evidence present", str(self.answered_with_evidence), "correct"),
            ("abstained, evidence absent", str(self.abstained_without_evidence), "correct"),
            ("abstained, evidence present", str(self.abstained_with_evidence),
             "wrong — it had the answer"),
            ("answered, evidence absent", str(self.answered_without_evidence),
             "wrong — answered anyway"),
            ("unparseable responses", str(self.parse_failures),
             f"of {answered + self.parse_failures} attempts"),
        ]


def _score_one(item: BenchItem, answer: Answer, context_ids: set[str],
               retrieved_evidence: bool) -> GenerationResult:
    sufficient = set(item.all_evidence)
    citations = good = 0
    grounded = 0
    cited_gold = False
    for claim in answer.claims:
        if claim.grounded:
            grounded += 1
        for vid, span in claim.spans.items():
            citations += 1
            if vid in sufficient:
                cited_gold = True
            if vid in context_ids and span is not None:
                good += 1
    return GenerationResult(
        item_id=item.id, retrieved_evidence=retrieved_evidence,
        abstained=answer.abstained, parse_failed=answer.parse_failed,
        claims=len(answer.claims), grounded_claims=grounded,
        citations=citations, good_citations=good, cited_gold=cited_gold,
    )


def score_generation(retriever: Retriever, generator, items: list[BenchItem], *,
                     context_k: int = 16) -> GenerationReport:
    """Retrieve, generate, and score the answer against evidence already in the benchmark."""
    report = GenerationReport(n=len(items))
    store = retriever.store
    for item in items:
        trace = retriever.retrieve(item.query, as_of=item.as_of, scope=item.scope)
        excerpts = excerpts_for(store, trace, limit=context_k)
        context_ids = {vid for vid, _, _ in excerpts}
        answer = generator.answer(item.query, excerpts, as_of=item.as_of,
                                  scope=item.scope.describe())
        report.results.append(_score_one(
            item, answer, context_ids,
            retrieved_evidence=item.is_satisfied_by(list(context_ids))))
    return report
