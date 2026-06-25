"""The coding-agent memory bench (Fase B) pins the vertical's properties end-to-end: a revised
architecture decision surfaces its current value, prior bugs/failures resurface on related queries, and
forbidden actions are flagged without over-blocking benign ones. Deterministic — HashingEmbedder, no LLM."""
from __future__ import annotations

from eval.coding_bench import run


def test_coding_bench_passes() -> None:
    scores = run(verbose=False)
    assert scores["decision_currency"] == 1.0
    assert scores["repeated_mistake"] == 1.0
    assert scores["forbidden_accuracy"] == 1.0
