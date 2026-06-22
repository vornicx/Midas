"""The BEAM loader must carry the rubric-graded categories' grading data into Question.metadata.

summarization / instruction_following / preference_following ship NO reference answer string — only a
`rubric` (a list of "response should contain X" bullets) and, for summarization, an `ideal_summary`.
Those were being dropped, so the judged summarization A/B (`eval.summarization_ab`) had nothing to grade
against. This pins that they survive the loader. Skips when the (gitignored) parquet isn't present, so
CI without the dataset stays green."""
from pathlib import Path

import pytest

_DATA = Path(__file__).resolve().parent.parent / "data" / "beam" / "100K.parquet"
pytestmark = pytest.mark.skipif(
    not _DATA.exists(), reason="BEAM 100K parquet not present (gitignored dataset)"
)


def test_summarization_carries_rubric_and_keeps_no_reference_answer():
    from eval.datasets import beam

    ds = beam(tier="100K", max_conversations=2)
    summ = [q for s in ds.samples for q in s.questions if q.category == "summarization"]
    assert summ, "expected summarization questions in BEAM-100K"
    for q in summ:
        # Rubric-graded: still no gold string (so answer_dumb/recall@k behaviour is unchanged)...
        assert q.answer is None
        # ...but the rubric bullets and the reference summary now ride along for a rubric judge.
        rubric = q.metadata.get("rubric")
        assert rubric and all(isinstance(b, str) and b.strip() for b in rubric)
        assert q.metadata.get("ideal_summary")


def test_reference_answer_categories_are_unaffected():
    from eval.datasets import beam

    ds = beam(tier="100K", max_conversations=2)
    # Categories that DO ship an answer keep it — the metadata addition is purely additive.
    answered = [
        q for s in ds.samples for q in s.questions
        if q.category in {"knowledge_update", "temporal_reasoning", "information_extraction"}
    ]
    assert answered and all(q.answer for q in answered)
