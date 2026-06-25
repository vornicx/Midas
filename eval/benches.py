"""The Midas agent-memory bench suite — every governed-memory benchmark as one report card.

Retrieval benchmarks (LongMemEval / LoCoMo / BEAM) measure *"can you find the fact"*. These measure what
a **governed** memory layer must guarantee for an agent to act on it safely — and nobody else publishes
them. All deterministic, $0, no LLM, reproducible. This is the standard we propose for evaluating agent
memory **beyond recall@k**; run it on Midas, or adapt the scenarios to any memory system.

    uv run python -m eval.benches
"""
from __future__ import annotations

from . import coding_bench, continuity, memory_safety


def run(verbose: bool = True) -> dict:
    cont = continuity.run(verbose=False)
    safe = memory_safety.run(verbose=False)
    code = coding_bench.run(verbose=False)

    rows = [
        ("Agent Continuity", "action_safety", cont["action_safety"], "→1.00"),
        ("", "decision_adherence", cont["decision_adherence"], "→1.00"),
        ("", "repeated_mistake", cont["repeated_mistake"], "→1.00"),
        ("Memory-Safety", "ASR (attack-success)", safe["ASR"], "→0.00"),
        ("", "benign_pass", safe["benign_pass"], "→1.00"),
        ("Coding-agent", "decision_currency", code["decision_currency"], "→1.00"),
        ("", "repeated_mistake", code["repeated_mistake"], "→1.00"),
        ("", "forbidden_accuracy", code["forbidden_accuracy"], "→1.00"),
    ]
    if verbose:
        print("=== Midas agent-memory bench suite (governed memory, deterministic, $0) ===\n")
        print(f"{'bench':<18}{'metric':<22}{'score':>6}{'  target':>9}")
        for bench, metric, score, target in rows:
            print(f"{bench:<18}{metric:<22}{score:>6.2f}{target:>9}")
        # green when every attack is stopped and every other metric is perfect
        green = safe["ASR"] == 0.0 and all(
            v == 1.0 for d in (cont, code) for v in d.values()
        ) and safe["benign_pass"] == 1.0
        print(f"\nVERDICT: {'all green ✓' if green else 'see failing rows above'}")
    return {"continuity": cont, "memory_safety": safe, "coding": code}


if __name__ == "__main__":
    run()
