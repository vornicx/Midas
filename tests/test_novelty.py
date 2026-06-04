"""Novelty-vs-store importance (no LLM) — score a turn by how NEW it is vs what's already stored.

The content heuristic scores a turn in isolation and can't tell a first mention from its tenth repeat.
Novelty-vs-store adds the missing context: a turn dissimilar to everything in memory is informative;
a near-restatement of a known fact is not. Blended with content salience, it lets a *new* fact outrank
a *repeated* one — sharpening what selective forgetting keeps.
"""
from __future__ import annotations

from midas import ContentImportance, Memory


class FixedEmbedder:
    """'alpha …' all map to one vector (duplicates); 'beta' is orthogonal."""

    dim = 2

    def embed(self, text: str) -> list[float]:
        return {"alpha": [1.0, 0.0], "alpha again": [1.0, 0.0], "beta": [0.0, 1.0]}[text]


def test_first_mention_outranks_a_repeat() -> None:
    m = Memory(embedder=FixedEmbedder(), novelty_weight=1.0)  # pure novelty
    first = m.remember("alpha")        # empty store -> wholly novel
    repeat = m.remember("alpha again")  # exact duplicate of the first -> not novel
    assert first.importance > repeat.importance
    assert repeat.importance == 1


def test_orthogonal_turns_are_both_novel() -> None:
    m = Memory(embedder=FixedEmbedder(), novelty_weight=1.0)
    a = m.remember("alpha")
    b = m.remember("beta")  # orthogonal to alpha -> still novel
    assert a.importance == b.importance == 5


def test_novelty_off_by_default_leaves_importance_unchanged() -> None:
    m = Memory(embedder=FixedEmbedder())  # novelty_weight defaults to 0
    assert m.remember("alpha").importance == 1
    assert m.remember("alpha again").importance == 1  # no scorer, no novelty -> default 1


def test_blended_with_content_a_repeat_scores_lower() -> None:
    # Default hashing embedder: identical text -> identical vector -> the repeat reads as a duplicate.
    m = Memory(novelty_weight=0.5, importance_scorer=ContentImportance())
    fact = "My flight to Tokyo is on September 14 at 3pm."
    first = m.remember(fact)
    repeat = m.remember(fact)
    assert first.importance > repeat.importance  # novelty pulls the redundant restatement down


def test_explicit_importance_is_not_overridden_by_novelty() -> None:
    m = Memory(embedder=FixedEmbedder(), novelty_weight=1.0)
    m.remember("alpha", importance=2)
    repeat = m.remember("alpha again", importance=2)  # caller's value wins, novelty does not apply
    assert repeat.importance == 2


def test_novelty_weight_is_clamped_to_unit_range() -> None:
    m = Memory(embedder=FixedEmbedder(), novelty_weight=5.0)  # clamps to 1.0
    assert m._novelty_weight == 1.0


def test_remember_many_sees_earlier_items_in_the_same_batch() -> None:
    m = Memory(embedder=FixedEmbedder(), novelty_weight=1.0)
    recs = m.remember_many([{"content": "alpha"}, {"content": "alpha again"}])
    # The second item is a duplicate of the first *within the same batch* -> lower importance.
    assert recs[0].importance > recs[1].importance
