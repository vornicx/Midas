"""The agent-memory bench suite (the public-standard wedge) stays all-green: every governance/safety/coding
property holds in one run. Deterministic — no LLM."""
from __future__ import annotations

from eval.benches import run


def test_bench_suite_all_green() -> None:
    out = run(verbose=False)
    assert out["continuity"] == {"action_safety": 1.0, "decision_adherence": 1.0, "repeated_mistake": 1.0}
    assert out["memory_safety"] == {"ASR": 0.0, "benign_pass": 1.0}
    assert out["coding"] == {"decision_currency": 1.0, "repeated_mistake": 1.0, "forbidden_accuracy": 1.0}
