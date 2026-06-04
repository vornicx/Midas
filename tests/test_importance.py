"""No-LLM content importance scoring — the salience signal forgetting needs on raw turns.

The scorer must separate fact-bearing turns (names, numbers, dense content) from chit-chat/filler, so
that selective forgetting protects the answer-bearing tier instead of decaying by age alone.
"""
from __future__ import annotations

from midas import ContentImportance, Memory


def test_backchannel_filler_scores_minimum() -> None:
    score = ContentImportance()
    for filler in ["haha yeah", "ok sounds good", "lol nice", "sure", "thanks!"]:
        assert score(filler) == 1, filler


def test_concrete_fact_outranks_filler() -> None:
    score = ContentImportance()
    fact = score("I adopted a dog named Max in 2019 in Chicago.")
    filler = score("yeah that sounds really good to me")
    assert fact >= 3
    assert fact > filler


def test_numbers_and_named_entities_lift_importance() -> None:
    score = ContentImportance()
    # A name (proper noun) + a number/date should both register as specifics.
    assert score("My flight to Tokyo leaves on September 14 at 3pm.") >= 3
    assert score("The budget ceiling is 2000 euros per month.") >= 3
    # A long but generic musing without specifics stays modest.
    assert score("i was just thinking about how nice it is to relax sometimes") <= 2


def test_scores_are_bounded_1_to_5() -> None:
    score = ContentImportance()
    dense = score("Dr. Mara Velez relocated the Polaris launch from Berlin to Lisbon on March 3 2026 "
                  "after the PostgreSQL migration finished ahead of the 200ms latency deadline.")
    assert 1 <= dense <= 5
    assert score("") == 1


def test_memory_derives_importance_when_omitted() -> None:
    mem = Memory(importance_scorer=ContentImportance())
    fact = mem.remember("I adopted a dog named Max in 2019.", kind="chat")  # no importance passed
    filler = mem.remember("haha yeah", kind="chat")
    assert fact.importance > filler.importance
    assert filler.importance == 1


def test_explicit_importance_overrides_scorer() -> None:
    mem = Memory(importance_scorer=ContentImportance())
    rec = mem.remember("haha yeah", kind="chat", importance=5)  # caller's value wins
    assert rec.importance == 5


def test_no_scorer_keeps_default_importance() -> None:
    mem = Memory()  # no scorer
    rec = mem.remember("I adopted a dog named Max in 2019.", kind="chat")
    assert rec.importance == 1  # unchanged default behaviour


def test_remember_many_derives_importance() -> None:
    mem = Memory(importance_scorer=ContentImportance())
    recs = mem.remember_many([
        {"content": "ok", "kind": "chat"},
        {"content": "My passport number is 12345678 and it expires in 2030.", "kind": "chat"},
    ])
    assert recs[0].importance == 1
    assert recs[1].importance > recs[0].importance
