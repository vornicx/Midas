"""Integrity checks for the adversarial conflicts-v1 dataset (near-duplicates + temporal
conflicts). These pin the trap structure the published numbers depend on."""
from __future__ import annotations

from eval.datasets import conflicts


def test_conflicts_dataset_shape() -> None:
    dataset = conflicts()

    assert dataset.name == "conflicts-v1"
    sample = dataset.samples[0]
    event_ids = [e.id for e in sample.events]
    assert len(event_ids) == len(set(event_ids))
    # Every gold id must exist in the event stream.
    for q in sample.questions:
        for gid in q.gold_event_ids:
            assert gid in event_ids, f"{q.id} references missing event {gid}"


def test_conflicts_current_questions_carry_the_outdated_value() -> None:
    sample = conflicts().samples[0]
    contents = " ".join(e.content for e in sample.events)

    for q in sample.questions:
        if q.category in ("current", "confusable"):
            assert q.stale_answer, f"{q.id} must annotate the trap value"
            # Both the gold answer and the trap value actually appear in the event stream —
            # otherwise ctx_stale/ctx_contradict cannot measure anything.
            assert q.answer.lower() in contents.lower()
            assert q.stale_answer.lower() in contents.lower()


def test_conflicts_temporal_traps_repeat_the_old_value_more_than_the_new() -> None:
    """The adversarial property: the OUTDATED value is stated more often than the update."""
    sample = conflicts().samples[0]
    for q in sample.questions:
        if q.category != "current":
            continue
        stale_mentions = sum(
            q.stale_answer.lower() in e.content.lower() for e in sample.events
        )
        current_mentions = sum(q.answer.lower() in e.content.lower() for e in sample.events)
        assert stale_mentions >= current_mentions, (
            f"{q.id}: old value must be at least as prominent as the new one"
        )


def test_conflicts_confusable_pairs_are_both_true() -> None:
    """Entity-confusable questions come in pairs: both facts are simultaneously true, so each
    question's gold is the other's trap. Wrongly superseding one with the other fails a question."""
    sample = conflicts().samples[0]
    confusable = {q.id: q for q in sample.questions if q.category == "confusable"}
    assert len(confusable) >= 4
    for q in confusable.values():
        partner = [
            p
            for p in confusable.values()
            if p.id != q.id and p.answer == q.stale_answer and p.stale_answer == q.answer
        ]
        assert partner, f"{q.id} has no mirrored confusable partner"
