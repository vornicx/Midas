from __future__ import annotations

import pytest

from midas import Armorer, Memory, decide_memory_use


def test_memory_records_default_to_observation_provenance() -> None:
    rec = Memory().remember("The deploy target is staging.", kind="fact")

    assert rec.provenance == "observation"


def test_guard_allows_planning_but_blocks_external_action_without_user_confirmation() -> None:
    rec = Memory().remember("Use the staging deploy target.", kind="fact", provenance="observation")

    planning = decide_memory_use([rec], intended_use="planning")
    action = decide_memory_use([rec], intended_use="external_action")

    assert planning.allowed is True
    assert action.allowed is False
    assert action.blocked_ids == (rec.id,)


def test_guard_allows_external_action_from_user_confirmation() -> None:
    rec = Memory().remember(
        "User confirmed: deploy to staging.",
        kind="constraint",
        provenance="user_confirmation",
        actor="user",
    )

    decision = decide_memory_use([rec], intended_use="external_action", acting_agent="agent-b")

    assert decision.allowed is True
    assert decision.evidence[0].provenance == "user_confirmation"


def test_guard_allows_external_action_when_mixed_recall_includes_confirmation() -> None:
    mem = Memory()
    stale = mem.remember("Deploy target was staging last week.", kind="fact", provenance="observation")
    confirmed = mem.remember(
        "User confirmed: deploy to staging.",
        kind="constraint",
        provenance="user_confirmation",
        actor="user",
    )

    decision = decide_memory_use([stale, confirmed], intended_use="external_action")

    assert decision.allowed is True
    assert decision.blocked_ids == (stale.id,)
    assert [e.id for e in decision.evidence] == [confirmed.id]


def test_guard_blocks_internal_plans_as_answer_evidence() -> None:
    rec = Memory().remember("Maybe switch the database to SQLite.", kind="note", provenance="planning")

    decision = decide_memory_use([rec], intended_use="answer")

    assert decision.allowed is False


def test_guard_blocks_superseded_confirmation_for_destructive_action() -> None:
    # Adversarial currency case: a memory the user confirmed WHEN CURRENT, later revised. The stale
    # belief must not justify an action on its own — the guard's defense-in-depth against acting on an
    # outdated confirmation (e.g. "delete the X bucket" after X was migrated away).
    mem = Memory()
    rec = mem.remember(
        "User confirmed: delete the staging bucket.",
        kind="constraint",
        provenance="user_confirmation",
        actor="user",
    )
    rec.superseded_by = "newer-belief-id"  # belief was revised after confirmation

    destructive = decide_memory_use([rec], intended_use="destructive_action")
    assert destructive.allowed is False
    assert destructive.blocked_ids == (rec.id,)
    # Answering from a stale belief is wrong too...
    assert decide_memory_use([rec], intended_use="answer").allowed is False
    # ...but planning may still consider the superseded history.
    assert decide_memory_use([rec], intended_use="planning").allowed is True


def test_memory_guard_reliance_recalls_and_checks_policy() -> None:
    mem = Memory()
    mem.remember("The billing limit is 200 dollars.", kind="constraint", provenance="observation")

    decision = mem.guard_reliance("billing limit", intended_use="destructive_action")

    assert decision.allowed is False
    assert decision.evidence


def test_armorer_stamps_actor_session_and_source() -> None:
    stamp = Armorer(actor="agent-a", session="s1").remember_kwargs(
        "action", tool="shell", source="tool:shell"
    )

    assert stamp["provenance"] == "action"
    assert stamp["actor"] == "agent-a"
    assert stamp["source"] == "tool:shell"
    assert stamp["metadata"] == {"session": "s1", "tool": "shell"}


def test_invalid_provenance_fails_fast() -> None:
    with pytest.raises(ValueError, match="invalid memory provenance"):
        Memory().remember("bad stamp", provenance="rumor")  # type: ignore[arg-type]
