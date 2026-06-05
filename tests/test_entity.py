"""Entity-grounded abstention (no LLM) — offline validation of the Calibrated-frontier candidate.

The diagnosed failure: the reader confabulates an answer drawn from a distractor about a DIFFERENT
entity than the question asks about. The entity check abstains when the answer's source turn does not
mention the question's focus entity. These cases pin the mechanism on the diagnosed failure modes
(crafted — the end-to-end win still needs a capable reader; the local 1B model doesn't confabulate).
"""
from __future__ import annotations

import pytest

from midas.entity import entities, entity_grounded, question_focus

# (question, source, is_confab) — a confab (wrong-entity source) must read as NOT grounded.
CASES = [
    ("What is the name of the user's hamster?", "I have a cat named Luna.", True),
    ("What is the name of the user's hamster?", "My hamster is named Coco.", False),
    ("What city does the user live in?", "I live in Barcelona.", False),
    ("What is the user's favorite color?", "My favorite food is sushi.", True),
    ("Where does the user work?", "I work at Acme Corp.", False),
    ("What is the user's dog's name?", "The meeting is on Tuesday.", True),
    ("What is the user's phone number?", "My phone number is 555-1234.", False),
    ("What breed is the user's dog?", "I have a cat named Luna.", True),
]


@pytest.mark.parametrize("question,source,is_confab", CASES)
def test_entity_grounding_separates_confab_from_real(question, source, is_confab):
    grounded = entity_grounded(question, source)
    # confab -> should NOT be grounded (abstain); real answer -> grounded (keep)
    assert grounded == (not is_confab)


def test_attribute_words_are_not_the_entity():
    # "favorite" must not count as the shared entity (it recurs across confab pairs).
    assert "favorite" not in question_focus("What is the user's favorite color?")
    assert question_focus("What is the user's favorite color?") == {"color"}


def test_entities_drops_scaffolding_keeps_nouns():
    e = entities("My hamster is named Coco")
    assert "hamster" in e and "coco" in e
    assert "is" not in e and "my" not in e


def test_entity_grounds_answer_finds_source_and_flags_wrong_entity():
    from eval.metrics import _entity_grounds_answer

    q = "What is the name of the user's hamster?"
    turns = ["I have a cat named Luna.", "My hamster is named Coco.", "We met on Tuesday."]
    assert not _entity_grounds_answer(q, "Luna", "", top_texts=turns)  # confab from the cat turn -> abstain
    assert _entity_grounds_answer(q, "Coco", "", top_texts=turns)      # real answer from the hamster turn
    assert _entity_grounds_answer(q, "I don't know", "", top_texts=turns)  # already abstaining -> leave it
