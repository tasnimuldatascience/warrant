"""Mining training triples, and the split discipline that makes the fine-tune reportable.

The load-bearing test is `test_no_held_out_section_reaches_the_training_set`. Sections are
hashed to a side by `assign_split`; if one leaks, every downstream retrieval number is
measured on text the encoder was trained on, every number still computes, and nothing
anywhere reports an error. That failure is invisible by construction, so it is asserted here
rather than trusted.

Offline and CPU-only. The one test that actually trains is marked ``neural`` and is off by
default, so the ordinary suite neither downloads weights nor imports torch.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from warrant.eval.bench import assign_split, mine_temporal
from warrant.index.store import Chunk, Store
from warrant.train.mine import (
    AMENDMENT,
    LEXICAL,
    SIBLING,
    Triple,
    batches,
    document_text,
    mine,
    triples_from_items,
)

T0 = "2026-01-01T00:00:00+00:00"
HORIZON = "2026-08-26"

#: Fixed by ``assign_split``'s hash of the section id, not chosen for looks. Asserted below
#: so a change to the split function fails here rather than silently making these tests
#: assert nothing.
DEV_SECTION = "890.301"
TEST_SECTION = "315.803"

HEADING = "Agency action during probationary period"
OLD = ("(a) The agency shall utilize the probationary period as fully as possible to "
       "determine the fitness of the employee for continued employment.")
NEW = ("(a) The agency shall utilize the probationary period as fully as possible to "
       "determine the fitness of the employee for continued employment. The agency must "
       "notify supervisors three months prior to expiration of the probationary period.")
SIBLING_TEXT = ("(b) An agency shall notify the employee in writing of the reasons for "
                "termination and the effective date of the separation action.")
#: The held-out section needs its own vocabulary. Two sections carrying identical prose
#: produce a byte-identical query with two different golds, and ``_drop_ambiguous`` -- quite
#: rightly -- deletes both, leaving a test that asserts nothing about an empty list.
OTHER_OLD = ("(a) Enrollment in a health benefits plan continues during a period of "
             "leave without pay for not more than 365 days.")
OTHER_NEW = ("(a) Enrollment in a health benefits plan continues during a period of "
             "leave without pay for not more than 365 days, unless the enrollee elects "
             "in writing to terminate the enrollment.")


@pytest.fixture
def store() -> Store:
    with Store(":memory:") as s:
        yield s


def add(store: Store, section: str, anchor: str, text: str,
        valid_from: str, valid_to: str | None) -> None:
    store.add([Chunk(chunk_id=f"{section}#{anchor}", section_id=section, title=5,
                     part=section.split(".")[0], anchor=anchor,
                     heading=HEADING, text=text,
                     valid_from=valid_from, valid_to=valid_to,
                     source_snapshot=valid_from, config_hash="t")], system_from=T0)


def amended(store: Store, section: str, old: str = OLD, new: str = NEW) -> None:
    """One paragraph amended once, plus an unamended sibling paragraph."""
    add(store, section, "a", old, "2017-01-01", "2020-11-16")
    add(store, section, "a", new, "2020-11-16", None)
    add(store, section, "b", SIBLING_TEXT, "2017-01-01", None)


def test_the_split_of_the_fixture_sections_is_what_the_tests_assume():
    assert assign_split(DEV_SECTION) == "dev"
    assert assign_split(TEST_SECTION) == "test"


def test_an_amendment_yields_triples(store: Store):
    amended(store, DEV_SECTION)
    mined = mine(store, horizon=HORIZON, split="dev")
    assert mined.triples
    assert mined.counts[AMENDMENT] > 0
    assert all(t.section_id == DEV_SECTION for t in mined.triples)


def test_the_hard_negative_is_genuinely_the_other_version(store: Store):
    """Not a paraphrase and not a sampled unrelated paragraph: the same paragraph on the
    other side of a real amendment, which is the hardest negative a retriever can be given."""
    amended(store, DEV_SECTION)
    mined = mine(store, horizon=HORIZON, split="dev")
    hard = mined.of_kind(AMENDMENT)
    assert hard
    for t in hard:
        assert {t.positive, t.negative} == {document_text(HEADING, OLD),
                                            document_text(HEADING, NEW)}


def test_positives_are_the_string_the_dense_index_embeds():
    """``dense.build`` prefixes the heading, because a paragraph reading "(b) The period may
    be extended once." is unretrievable without it. Training on the bare text would optimise
    a document form the index never contains."""
    assert document_text("Heading", "Body") == "Heading. Body"
    assert document_text(None, "Body") == "Body"


def test_no_held_out_section_reaches_the_training_set(store: Store):
    amended(store, DEV_SECTION)
    amended(store, TEST_SECTION, OTHER_OLD, OTHER_NEW)
    mined = mine(store, horizon=HORIZON, split="dev")

    assert mined.triples
    assert {t.section_id for t in mined.triples} == {DEV_SECTION}
    assert all(assign_split(t.section_id) == "dev" for t in mined.triples)
    # Negatives too, not only positives: a held-out section's paragraph used as a negative
    # is still that section's text inside the training set.
    held_out = "health benefits plan"
    for t in mined.triples:
        assert held_out not in t.positive
        assert held_out not in t.negative


def test_a_forged_split_label_does_not_override_the_hash(store: Store):
    """``BenchItem.split`` is data; ``assign_split`` is the rule. Only the rule is trusted."""
    amended(store, TEST_SECTION)
    items = [replace(i, split="dev") for i in mine_temporal(store, horizon=HORIZON)]
    assert items and all(i.split == "dev" for i in items)
    mined = triples_from_items(store, items, split="dev")
    assert mined.triples == []


def test_in_domain_negatives_are_mined_beside_the_amendment(store: Store):
    amended(store, DEV_SECTION)
    mined = mine(store, horizon=HORIZON, split="dev")
    assert mined.counts[SIBLING] > 0, "the section's other paragraph should be a negative"
    assert any(SIBLING_TEXT in t.negative for t in mined.of_kind(SIBLING))
    assert mined.counts[LEXICAL] >= 0    # BM25 may find nothing else in a two-section store


def test_identical_triples_are_emitted_once(store: Store):
    amended(store, DEV_SECTION)
    items = mine_temporal(store, horizon=HORIZON)
    once = triples_from_items(store, items, split="dev")
    twice = triples_from_items(store, list(items) + list(items), split="dev")
    assert len(twice.triples) == len(once.triples)


def test_an_item_with_no_distractor_is_skipped(store: Store):
    """A pure addition or deletion has no counterpart version, so it can teach nothing about
    near-duplicates and is dropped whole rather than kept on its weaker negatives."""
    amended(store, DEV_SECTION)
    items = [replace(i, distractors=[]) for i in mine_temporal(store, horizon=HORIZON)]
    mined = triples_from_items(store, items, split="dev")
    assert mined.triples == []
    assert mined.items_without_distractor == len(items)


def test_the_two_sides_of_one_amendment_are_counted_as_contradictory(store: Store):
    """Both sides share a query and each is the other's gold. A bi-encoder never sees
    ``as_of``, so that pair of triples cancels; the count is reported rather than hidden."""
    amended(store, DEV_SECTION)
    mined = mine(store, horizon=HORIZON, split="dev")
    assert mined.contradictory_queries == 1


def test_a_batch_never_holds_two_examples_for_one_paragraph():
    """In-batch negatives make every other row's positive a negative for this row, so a
    repeated paragraph inside one batch labels the correct passage as wrong. On this data
    that is not a rare accident: the before and after sides of one amendment share a query."""
    triples = [Triple(query=f"q{p}", positive=f"pos {p}", negative=f"neg {p}-{n}",
                      section_id="890.301", chunk_id=f"890.301#{p}", kind=AMENDMENT)
               for p in "abc" for n in range(4)]
    plan = batches(triples, batch_size=3, seed=0)
    assert plan
    for batch in plan:
        assert len(batch.queries) == len(batch.positives) == len(batch.negatives)
        assert len(set(batch.positives)) == len(batch.positives)
    assert sum(len(b) for b in plan) == len(triples)


@pytest.mark.neural
def test_finetune_writes_a_reproducible_sidecar(store: Store, tmp_path):
    """Downloads weights and trains; excluded from the default suite by the marker."""
    from warrant.train.finetune import TrainSpec, finetune, load_metadata

    amended(store, DEV_SECTION)
    mined = mine(store, horizon=HORIZON, split="dev")
    result = finetune(mined, tmp_path / "encoder",
                      spec=TrainSpec(epochs=1, batch_size=4, seed=0, device="cpu"))
    meta = load_metadata(tmp_path / "encoder")
    assert result.steps > 0
    assert meta["seed"] == 0 and meta["train_split"] == "dev"
    assert meta["triples"] == result.triples
    assert meta["base_model"]
