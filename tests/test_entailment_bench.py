"""Tests for `warrant.eval.entailment` -- the loader and scorer behind eval-007.

The interesting tests are the loader's failure modes, because every one of them is a way the
benchmark could keep reporting a number while measuring something else: an anchor that no
longer resolves and is quietly dropped, two strata averaged into a mixture nobody chose, a
label outside the three the confusion matrix has rows for.

Scoring is exercised with injected logits rather than the checkpoint. The arithmetic --
micro, macro, the confusion matrix, the leave-one-section-out temperature -- is the part a
reviewer reads and the part most likely to be wrong in a way no metric would catch, and it
runs on a clone that has never downloaded a model. The one test that loads weights carries
`@pytest.mark.neural`, which the default `pytest` invocation excludes, and additionally skips
when the weights are not already cached so that `-m neural` on a fresh clone reports skipped
rather than reaching the network mid-suite.
"""

from __future__ import annotations

import pytest
import yaml

from warrant.config import Config
from warrant.eval.entailment import (
    ADVERSARIAL,
    DEFAULT_PATH,
    GENERATOR,
    LABELS,
    BenchmarkError,
    UnresolvedEvidence,
    load,
    score,
)
from warrant.index.store import Chunk, Store
from warrant.verify.entail import CALIBRATION_TEMPERATURE, DEFAULT_MODEL

T0 = "2026-01-01T00:00:00+00:00"

PREMISE = ("(a) An agency must comply with an order to reinstate issued by OPM under this "
           "section as promptly as possible, but not more than 30 calendar days from the "
           "date of the order.")
OTHER = ("(b) A competitive area must be defined solely in terms of the agency's "
         "organizational unit(s) and geographical location.")


def chunk(chunk_id: str, text: str, valid_from: str = "2017-01-01",
          valid_to: str | None = None) -> Chunk:
    section, anchor = chunk_id.split("#")
    return Chunk(chunk_id=chunk_id, section_id=section, title=5, part=section.split(".")[0],
                 anchor=anchor, heading="", text=text, valid_from=valid_from,
                 valid_to=valid_to, source_snapshot=valid_from, config_hash="t")


@pytest.fixture
def store() -> Store:
    with Store(":memory:") as s:
        s.add([chunk("317.703#f", PREMISE), chunk("351.402#b", OTHER)], system_from=T0)
        yield s


def write(tmp_path, entries: list[dict]):
    path = tmp_path / "entailment.yaml"
    path.write_text(yaml.safe_dump(entries, allow_unicode=True), encoding="utf-8")
    return path


def item(**kw) -> dict:
    base = {"id": "gen-0", "stratum": GENERATOR, "evidence": "317.703#f",
            "as_of": "2017-01-01", "label": "entail",
            "claim": "An agency has at most 30 calendar days to comply."}
    base.update(kw)
    return base


# -- the loader -------------------------------------------------------------------


def test_evidence_is_resolved_to_a_version_id_and_the_stores_own_text(tmp_path, store):
    """The file names `section#anchor`; the premise comes from the store, never the file."""
    pairs = load(write(tmp_path, [item()]), store=store)
    assert len(pairs) == 1
    assert pairs[0].version_id == "317.703#f@2017-01-01"
    assert pairs[0].premise == PREMISE
    assert pairs[0].section_id == "317.703"


def test_an_unresolvable_anchor_is_an_error_not_a_skip(tmp_path, store):
    """The failure this guards is silence, not a crash.

    A dropped pair keeps its line in the file, so the set shrinks while the `n` a reader
    quotes does not move -- and the shrinking is exactly toward whichever pairs survived a
    rebuild, which is not a random subset.
    """
    path = write(tmp_path, [item(evidence="317.703#nonexistent")])
    with pytest.raises(UnresolvedEvidence, match="317.703#nonexistent"):
        load(path, store=store)


def test_evidence_that_was_amended_out_before_the_pairs_date_is_unresolvable(tmp_path):
    """Resolution is bitemporal: existing under some other date is not resolving."""
    with Store(":memory:") as s:
        s.add([chunk("317.703#f", PREMISE, valid_from="2017-01-01", valid_to="2020-01-01")],
              system_from=T0)
        path = write(tmp_path, [item(as_of="2024-06-01")])
        with pytest.raises(UnresolvedEvidence):
            load(path, store=s)


def test_a_pair_resolves_to_the_version_in_force_on_its_own_date(tmp_path):
    """Two versions of one paragraph: the label was written against one of them."""
    with Store(":memory:") as s:
        s.add([chunk("317.703#f", "30 calendar days", "2017-01-01", "2020-01-01"),
               chunk("317.703#f", "45 calendar days", "2020-01-01")], system_from=T0)
        old, new = load(write(tmp_path, [item(id="gen-0", as_of="2018-06-01"),
                                         item(id="gen-1", as_of="2024-06-01")]), store=s)
    assert old.premise == "30 calendar days"
    assert new.premise == "45 calendar days"


def test_a_stratum_outside_the_two_is_rejected(tmp_path, store):
    """A third stratum is not scored by `score`, so it would silently vanish from every
    table while still occupying a line in the file."""
    path = write(tmp_path, [item(stratum="synthetic")])
    with pytest.raises(BenchmarkError, match="synthetic"):
        load(path, store=store)


@pytest.mark.parametrize("label", ["E", "supported", "entails", ""])
def test_a_label_outside_the_three_is_rejected(tmp_path, store, label):
    """The confusion matrix has three rows. A fourth label has nowhere to be counted, and
    the abbreviations the results doc uses in prose are not the labels on disk."""
    with pytest.raises(BenchmarkError):
        load(write(tmp_path, [item(label=label)]), store=store)


def test_a_duplicate_id_is_rejected(tmp_path, store):
    path = write(tmp_path, [item(id="gen-0"), item(id="gen-0", label="neutral")])
    with pytest.raises(BenchmarkError, match="duplicate"):
        load(path, store=store)


@pytest.mark.parametrize("field", ["id", "stratum", "evidence", "as_of", "label", "claim"])
def test_a_missing_field_is_rejected(tmp_path, store, field):
    entry = item()
    del entry[field]
    with pytest.raises(BenchmarkError):
        load(write(tmp_path, [entry]), store=store)


def test_provenance_is_optional_and_only_the_generator_stratum_carries_it(tmp_path, store):
    gen, adv = load(write(tmp_path, [
        item(id="gen-0", question="probation-length"),
        item(id="adv-0", stratum=ADVERSARIAL, label="contradict")]), store=store)
    assert gen.question == "probation-length"
    assert adv.question == ""


# -- scoring ----------------------------------------------------------------------


def logit_row(label: str, margin: float = 4.0) -> tuple[float, float, float]:
    """Logits whose argmax is `label`. `margin` sets how confident the head is."""
    return tuple(margin if name == label else 0.0 for name in LABELS)


def build(tmp_path, store, spec: list[tuple[str, str, str, str]], **kw):
    """`spec` is (id, stratum, evidence, gold) per pair."""
    return load(write(tmp_path, [item(id=i, stratum=s, evidence=e, label=g, **kw)
                                 for i, s, e, g in spec]), store=store)


def test_the_two_strata_are_scored_separately_and_never_averaged(tmp_path, store):
    """Pooling is the failure eval-007 names explicitly: the adversarial stratum's chosen
    class balance repairs a generator stratum that is at chance on two of three classes."""
    spec = [(f"gen-{i}", GENERATOR, "317.703#f", "entail") for i in range(4)]
    spec += [(f"adv-{i}", ADVERSARIAL, "351.402#b", "contradict") for i in range(4)]
    pairs = build(tmp_path, store, spec)
    logits = {p.id: logit_row("entail" if p.stratum == GENERATOR else "contradict")
              for p in pairs}
    logits["gen-0"] = logit_row("neutral")     # one generator miss, no adversarial miss

    report = score(pairs, logits=logits, temperature=1.0, samples=200)
    assert [s.stratum for s in report.strata] == [GENERATOR, ADVERSARIAL]
    assert report.stratum(GENERATOR).micro == pytest.approx(0.75)
    assert report.stratum(ADVERSARIAL).micro == pytest.approx(1.0)
    # The pooled figure -- 7/8 -- is what pooling would report, and it describes neither.
    assert not any(s.micro == pytest.approx(7 / 8) for s in report.strata)
    assert not hasattr(report, "micro")


def test_macro_diverges_from_micro_exactly_where_the_class_balance_is_skewed(tmp_path,
                                                                            store):
    """The finding of eval-007 section 1 in miniature: 86.8% micro over a 60.1% macro."""
    spec = [(f"gen-{i}", GENERATOR, "317.703#f", "entail") for i in range(8)]
    spec += [("gen-8", GENERATOR, "317.703#f", "neutral"),
             ("gen-9", GENERATOR, "317.703#f", "contradict")]
    pairs = build(tmp_path, store, spec)
    # Every prediction is `entail`: right on the eight the corpus supplies, wrong on the two
    # classes a verifier exists to catch.
    logits = {p.id: logit_row("entail") for p in pairs}

    s = score(pairs, logits=logits, temperature=1.0, samples=200).stratum(GENERATOR)
    assert s.micro == pytest.approx(0.8)
    assert s.macro == pytest.approx(1 / 3)
    assert [(c.label, c.correct, c.n) for c in s.per_class] == [
        ("entail", 8, 8), ("neutral", 0, 1), ("contradict", 0, 1)]


def test_macro_skips_a_class_the_stratum_does_not_contain(tmp_path, store):
    """Scoring an absent class as zero would punish the generator stratum for a class
    balance that is the observation rather than a design choice."""
    spec = [("gen-0", GENERATOR, "317.703#f", "entail"),
            ("gen-1", GENERATOR, "317.703#f", "neutral")]
    pairs = build(tmp_path, store, spec)
    logits = {p.id: logit_row(p.label) for p in pairs}
    s = score(pairs, logits=logits, temperature=1.0, samples=200).stratum(GENERATOR)
    assert s.macro == pytest.approx(1.0)
    assert s.per_class[2].n == 0


def test_confusion_rows_are_the_gold_counts_and_pool_by_addition(tmp_path, store):
    spec = [("gen-0", GENERATOR, "317.703#f", "entail"),
            ("gen-1", GENERATOR, "317.703#f", "neutral"),
            ("adv-0", ADVERSARIAL, "351.402#b", "contradict")]
    pairs = build(tmp_path, store, spec)
    logits = {"gen-0": logit_row("entail"), "gen-1": logit_row("entail"),
              "adv-0": logit_row("contradict")}
    report = score(pairs, logits=logits, temperature=1.0, samples=200)
    assert report.stratum(GENERATOR).confusion == ((1, 0, 0), (1, 0, 0), (0, 0, 0))
    assert report.stratum(ADVERSARIAL).confusion == ((0, 0, 0), (0, 0, 0), (0, 0, 1))
    assert report.pooled_confusion == ((1, 0, 0), (1, 0, 0), (0, 0, 1))
    assert report.contradiction_rates() == (1, 0, 0)


def test_a_pair_with_no_logits_is_an_error(tmp_path, store):
    pairs = build(tmp_path, store, [("gen-0", GENERATOR, "317.703#f", "entail"),
                                    ("gen-1", GENERATOR, "351.402#b", "entail")])
    with pytest.raises(BenchmarkError, match="gen-1"):
        score(pairs, logits={"gen-0": logit_row("entail")})


def test_the_temperature_is_fitted_leaving_the_pairs_own_section_out(tmp_path, store):
    """Otherwise a pair's `supported` verdict is read off a temperature fitted on its own
    label, and the calibration figure is in-sample."""
    spec = [(f"gen-{i}", GENERATOR, "317.703#f", "entail") for i in range(4)]
    spec += [(f"gen-{i}", GENERATOR, "351.402#b", "neutral") for i in range(4, 8)]
    pairs = build(tmp_path, store, spec)
    # One section is confidently right, the other confidently wrong, so the two held-out
    # fits cannot coincide unless the section is not actually being held out.
    logits = {p.id: logit_row("entail", margin=8.0) for p in pairs}

    report = score(pairs, logits=logits, samples=200)
    by_section = {p.section_id: report.temperatures[p.id] for p in pairs}
    assert len(set(by_section.values())) == 2
    assert report.fixed_temperature is None
    fixed = score(pairs, logits=logits, temperature=CALIBRATION_TEMPERATURE, samples=200)
    assert set(fixed.temperatures.values()) == {CALIBRATION_TEMPERATURE}


def test_the_contradiction_channel_is_what_alignment_cannot_do(tmp_path, store):
    """A flipped claim reuses the premise's vocabulary, so overlap confirms it as supported.

    This is the whole case for the module, and it is a property of the aligner rather than a
    tunable: a contradiction has *more* overlap with its premise than a paraphrase does.
    """
    flipped = ("An agency may take more than 30 calendar days from the date of the order to "
               "comply with an order to reinstate issued by OPM.")
    pairs = build(tmp_path, store, [("adv-0", ADVERSARIAL, "317.703#f", "contradict")],
                  claim=flipped)
    report = score(pairs, logits={"adv-0": logit_row("contradict")}, temperature=1.0,
                   samples=200)
    result = report.results[0]
    assert result.span is True                       # the aligner confirms it
    assert result.report == "contradicted"           # the model does not
    assert report.comparison(ADVERSARIAL).align_accuracy == 0.0
    assert report.comparison(ADVERSARIAL).nli_accuracy == 1.0


def test_scoring_is_deterministic(tmp_path, store):
    spec = [(f"gen-{i}", GENERATOR, "317.703#f", "entail") for i in range(3)]
    spec += [(f"adv-{i}", ADVERSARIAL, "351.402#b", "contradict") for i in range(3)]
    pairs = build(tmp_path, store, spec)
    logits = {p.id: logit_row(p.label) for p in pairs}
    a = score(pairs, logits=logits, samples=200, seed=7)
    b = score(pairs, logits=logits, samples=200, seed=7)
    assert [s.row() for s in a.strata] == [s.row() for s in b.strata]
    assert [c.row() for c in a.comparisons] == [c.row() for c in b.comparisons]


# -- the shipped file -------------------------------------------------------------


def corpus() -> Store | None:
    cfg = Config.load()
    return Store(cfg.store_path) if cfg.store_path.exists() else None


def test_the_shipped_benchmark_is_the_probe_set_eval_007_reports():
    """Its shape, asserted against the results doc rather than against itself.

    182 pairs over 91 sections in two strata, with the class balance the document tabulates.
    A pair that stops resolving raises inside `load`, so this also gates the corpus.
    """
    s = corpus()
    if s is None:
        pytest.skip("no corpus built")
    with s:
        pairs = load(DEFAULT_PATH, store=s)
    assert len(pairs) == 182
    assert len({p.section_id for p in pairs}) == 91
    gen = [p for p in pairs if p.stratum == GENERATOR]
    adv = [p for p in pairs if p.stratum == ADVERSARIAL]
    assert len(gen) == 129 and len({p.section_id for p in gen}) == 74
    assert len(adv) == 53 and len({p.section_id for p in adv}) == 20
    assert [sum(1 for p in gen if p.label == c) for c in LABELS] == [102, 24, 3]
    assert [sum(1 for p in adv if p.label == c) for c in LABELS] == [21, 10, 22]
    assert all(p.question for p in gen)


def test_the_file_addresses_evidence_by_anchor_and_never_carries_the_premise():
    """A premise pasted into the file, or a version id written by hand, stops tracking the
    corpus the moment the corpus is rebuilt -- the rot `human.yaml` avoids the same way."""
    raw = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
    assert {k for entry in raw for k in entry} <= {
        "id", "stratum", "evidence", "as_of", "label", "question", "claim"}
    assert all("#" in entry["evidence"] and "@" not in entry["evidence"] for entry in raw)


def weights_cached(model: str = DEFAULT_MODEL) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return try_to_load_from_cache(model, "config.json") is not None


@pytest.mark.neural
@pytest.mark.skipif(not weights_cached(), reason="NLI weights not in the local cache")
def test_reproduces_the_eval_007_headline():
    """A regression test against the published numbers, not an aspiration.

    These are the figures `results/eval-007-entailment.md` sections 1 and 2 report, and this
    re-ran them from the store rather than from the appendix. If the checkpoint, the corpus
    or a label moves, this fails and the document is wrong -- which is the entire reason the
    appendix was promoted to a file.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    s = corpus()
    if s is None:
        pytest.skip("no corpus built")
    with s:
        pairs = load(DEFAULT_PATH, store=s)
    report = score(pairs)

    gen, adv = report.stratum(GENERATOR), report.stratum(ADVERSARIAL)
    assert (gen.correct, gen.n) == (112, 129)          # 86.8%
    assert gen.macro == pytest.approx(0.601, abs=0.002)
    assert [(c.correct, c.n) for c in gen.per_class] == [(99, 102), (12, 24), (1, 3)]
    assert (adv.correct, adv.n) == (47, 53)            # 88.7%
    assert adv.macro == pytest.approx(0.890, abs=0.002)
    assert [(c.correct, c.n) for c in adv.per_class] == [(20, 21), (9, 10), (18, 22)]
    assert report.pooled_confusion == ((119, 4, 0), (6, 21, 7), (4, 2, 19))

    generator = report.comparison(GENERATOR)
    assert generator.delta.delta == pytest.approx(0.023, abs=0.002)
    assert not generator.delta.significant       # +2.3 at p = 0.55 is not a measurement
    adversarial = report.comparison(ADVERSARIAL)
    assert adversarial.delta.delta == pytest.approx(0.491, abs=0.002)
    assert adversarial.delta.significant
    # The aligner locates a span in every adversarial contradiction: the flipped claim reuses
    # the premise's vocabulary, which is precisely what overlap scores.
    assert all(r.span for r in report.results
               if r.pair.stratum == ADVERSARIAL and r.pair.label == "contradict")
