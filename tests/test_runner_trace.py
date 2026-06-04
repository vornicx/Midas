from __future__ import annotations

import json

from eval.datasets import longmemeval
from eval.metrics import llm_answer_and_judge, llm_judge_correct
from eval.runner import question_trace_by_adapter
from eval.schema import Question


class FakeLLM:
    def complete(self, system: str, user: str, *, max_tokens: int = 256, **kwargs) -> str:
        if "Context:" in user:
            return "Paris"
        return "YES"


def test_llm_answer_and_judge_returns_predicted_answer_and_score() -> None:
    question = Question("q1", "Which city?", "Paris", ["e1"])

    predicted, score = llm_answer_and_judge(FakeLLM(), "City: Paris", question)

    assert predicted == "Paris"
    assert score == 1.0
    assert llm_judge_correct(FakeLLM(), "City: Paris", question) == 1.0


def test_longmemeval_loader_can_sample_without_full_read(tmp_path) -> None:
    path = tmp_path / "longmemeval.json"
    items = []
    for idx in range(8):
        items.append(
            {
                "question_id": f"q{idx}",
                "question": f"Question {idx}?",
                "answer": f"answer {idx}",
                "question_type": "single-session-user",
                "haystack_session_ids": [f"s{idx}"],
                "answer_session_ids": [f"s{idx}"],
                "haystack_sessions": [
                    [
                        {"role": "user", "content": f"answer {idx}", "has_answer": True},
                        {"role": "assistant", "content": "ok", "has_answer": False},
                    ]
                ],
            }
        )
    path.write_text(json.dumps(items), encoding="utf-8")

    dataset = longmemeval(path=path, max_questions=3, seed=7)

    assert dataset.name == "longmemeval-s"
    assert len(dataset.samples) == 3
    assert all(len(sample.events) == 2 for sample in dataset.samples)
    assert all(sample.questions[0].gold_event_ids for sample in dataset.samples)


def test_question_trace_prints_predicted_answer_and_snippets() -> None:
    table = question_trace_by_adapter(
        {
            "midas": {
                "traces": [
                    {
                        "id": "q1",
                        "category": "fact",
                        "recall@k": 1.0,
                        "answer": 1.0,
                        "tokens": 12,
                        "predicted": "Paris",
                        "gold": ["e1"],
                        "retrieved": ["e1"],
                        "gold_snippets": ["e1: City: Paris"],
                        "retrieved_snippets": ["e1: City: Paris"],
                    }
                ]
            }
        }
    )

    assert "predicted" in table
    assert "gold_snippets" in table
    assert "retrieved_snippets" in table
    assert "| q1 | fact | 1.00 | 1.00 | 12.00 | Paris | e1 | e1 |" in table
    assert "e1: City: Paris" in table
