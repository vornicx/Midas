"""Structural importance — content salience + assertion/attribute structure (no LLM).

The bag-of-features content score can't tell a *fact* from a *question* that mentions the same salient
words. StructuralImportance boosts turns that assert a durable attribute (especially about the user) and
demotes questions/meta — so a memory worth keeping outranks a query that merely shares its topic.
"""
from __future__ import annotations

from midas import ContentImportance, StructuralImportance


def test_assertion_outranks_a_same_topic_question() -> None:
    s = StructuralImportance()
    assert s("My favorite food is sushi.") > s("Where is the best sushi in Tokyo?")
    assert s("The launch date is September 14.") > s("When is the launch date?")


def test_personal_facts_score_high() -> None:
    s = StructuralImportance()
    assert s("I'm allergic to penicillin.") >= 3
    assert s("My passport number is 12345678.") >= 3
    assert s("I prefer dark mode and concise answers.") >= 3


def test_questions_and_meta_score_low() -> None:
    s = StructuralImportance()
    assert s("what should we do this weekend?") == 1
    assert s("let me check the schedule") == 1
    assert s("sounds good, thanks!") == 1


def test_fixes_the_content_score_inversion() -> None:
    # ContentImportance scores the question >= the fact (both mention 'sushi'/'Tokyo'); structure flips it.
    c, s = ContentImportance(), StructuralImportance()
    fact, question = "My favorite food is sushi.", "Where is the best sushi in Tokyo?"
    assert c(question) >= c(fact)          # the content score's blind spot
    assert s(fact) > s(question)           # structure fixes it


def test_scores_bounded_and_callable() -> None:
    s = StructuralImportance()
    assert s("") == 1
    dense = s("My doctor Dr. Mara Velez set my surgery date for March 3 2026 in Lisbon.")
    assert 1 <= dense <= 5
