"""Auto-capture — Midas imposes the relevance policy so an agent can 'remember everything' safely.

`capture` must keep durable, relevant turns and reject trivia + duplicates, all with no LLM, reporting
why — so the agent (or MCP client) learns the bar instead of polluting memory.
"""
from __future__ import annotations

from midas import AGENT_MEMORY_INSTRUCTIONS, DEFAULT_POLICY, Memory, MemoryPolicy, policy_summary


def test_capture_stores_relevant_fact() -> None:
    m = Memory()  # default policy (floor 2); capture scores importance with ContentImportance
    r = m.capture("My account number is 4421 and the renewal is in 2027.", kind="fact")
    assert r.stored and r.record is not None and r.importance >= 2
    assert len(m.store.all()) == 1


def test_capture_skips_filler_below_floor() -> None:
    m = Memory()
    r = m.capture("haha yeah sounds good")
    assert not r.stored and "floor" in r.reason
    assert len(m.store.all()) == 0


def test_capture_skips_near_duplicate() -> None:
    m = Memory()
    m.capture("My passport number is 12345678 and it expires in 2030.", kind="fact")
    r = m.capture("My passport number is 12345678 and it expires in 2030.", kind="fact")
    assert not r.stored and "duplicate" in r.reason
    assert len(m.store.all()) == 1


def test_duplicate_capture_can_upgrade_provenance() -> None:
    m = Memory()
    m.capture("Deploy target is staging.", kind="constraint", provenance="observation")
    r = m.capture(
        "Deploy target is staging.",
        kind="constraint",
        provenance="user_confirmation",
        actor="user",
    )

    assert not r.stored and "provenance upgraded" in r.reason
    assert len(m.store.all()) == 1
    rec = m.store.all()[0]
    assert rec.provenance == "user_confirmation"
    assert m.guard_reliance("deploy target", intended_use="external_action").allowed


def test_capture_respects_accept_kinds() -> None:
    m = Memory(policy=MemoryPolicy(accept_kinds=("fact",)))
    r = m.capture("The budget ceiling is 2000 euros per month.", kind="chat")  # chat not accepted
    assert not r.stored and "not accepted" in r.reason


def test_capture_custom_floor_rejects_moderate_item() -> None:
    m = Memory(policy=MemoryPolicy(min_importance=5))  # only the most salient turns survive
    r = m.capture("My account number is 4421.", kind="fact")
    assert not r.stored and "floor" in r.reason


def test_capture_empty_is_skipped() -> None:
    assert not Memory().capture("   ").stored


def test_capture_per_call_policy_overrides_instance() -> None:
    m = Memory(policy=MemoryPolicy(min_importance=5))
    r = m.capture("Decision: we ship on Friday.", kind="note", policy=MemoryPolicy(min_importance=1))
    assert r.stored  # the per-call policy wins


def test_policy_and_instructions_surface_the_parameters() -> None:
    assert "capture" in AGENT_MEMORY_INSTRUCTIONS.lower() and "recall" in AGENT_MEMORY_INSTRUCTIONS.lower()
    summary = policy_summary(DEFAULT_POLICY)
    assert "importance >=" in summary and "duplicate" in summary
