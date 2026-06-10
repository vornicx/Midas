"""The dumb reader is the reader-independence ablation: a deterministic extractive 'reader' whose
score can only move with retrieval quality — it cannot reason, combine turns, or paraphrase. These
tests pin its mechanics so the published answer_dumb numbers stay reproducible."""
from __future__ import annotations

from eval.adapters.base import RetrievalResult
from eval.metrics import dumb_reader_score, extractive_answer, token_f1
from eval.schema import Question


def test_extractive_answer_picks_turn_with_most_question_overlap() -> None:
    result = RetrievalResult(
        context="",
        top_texts=[
            "We talked about the weather for a while.",
            "The launch date was moved to September 14.",
            "Someone brought churros for the team.",
        ],
    )
    question = Question("q1", "When is the launch date?", "September 14", ["e2"])

    assert extractive_answer(result, question) == "The launch date was moved to September 14."


def test_extractive_answer_falls_back_to_context_lines() -> None:
    result = RetrievalResult(
        context="# Memory\nThe project codename is Polaris.\nSmall talk about coffee.",
        top_texts=[],
    )
    question = Question("q1", "What is the project codename?", "Polaris", ["e1"])

    assert extractive_answer(result, question) == "The project codename is Polaris."


def test_extractive_answer_is_deterministic_on_ties() -> None:
    result = RetrievalResult(context="", top_texts=["unrelated alpha", "unrelated beta"])
    question = Question("q1", "Which database do we use?", "PostgreSQL", ["e1"])

    # No overlap anywhere -> keeps the first (highest-ranked) turn, every time.
    assert extractive_answer(result, question) == "unrelated alpha"


def test_token_f1_exact_partial_and_zero() -> None:
    assert token_f1("September 14", "September 14") == 1.0
    assert token_f1("the launch is September 14", "September 14") > 0.0
    assert token_f1("completely unrelated words", "September 14") == 0.0
    assert token_f1("", "September 14") == 0.0


def test_dumb_reader_score_full_credit_when_gold_in_extracted_turn() -> None:
    result = RetrievalResult(
        context="", top_texts=["The launch date was moved to September 14."]
    )
    question = Question("q1", "When is the launch date?", "September 14", ["e1"])

    assert dumb_reader_score(result, question) == 1.0


def test_dumb_reader_score_zero_when_retrieval_missed_the_fact() -> None:
    result = RetrievalResult(context="", top_texts=["We chatted about the weekend."])
    question = Question("q1", "When is the launch date?", "September 14", ["e1"])

    assert dumb_reader_score(result, question) == 0.0


def test_dumb_reader_score_none_for_unanswerable() -> None:
    result = RetrievalResult(context="", top_texts=["anything"])
    question = Question("q1", "What is the CEO's name?", None, [])

    assert dumb_reader_score(result, question) is None
