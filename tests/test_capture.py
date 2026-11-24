"""Auto-capture — Midas imposes the relevance policy so an agent can 'remember everything' safely.

`capture` must keep durable, relevant turns and reject trivia + duplicates, all with no LLM, reporting
why — so the agent (or MCP client) learns the bar instead of polluting memory.
"""
from __future__ import annotations

from midas import (
    AGENT_MEMORY_INSTRUCTIONS, DEFAULT_POLICY, HashingEmbedder, Memory, MemoryPolicy, policy_summary,
)


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


def test_capture_embeds_accepted_content_once_and_skips_text_only_rejections() -> None:
    class CountingEmbedder(HashingEmbedder):
        def __init__(self):
            super().__init__()
            self.texts = []

        def embed(self, text):
            self.texts.append(text)
            return super().embed(text)

    embedder = CountingEmbedder()
    mem = Memory(embedder=embedder, detect_standing=True)
    assert not mem.capture("haha yeah sounds good").stored
    assert not mem.capture("The deployment is pinned to version 16.", kind="chat",
                           policy=MemoryPolicy(accept_kinds=("fact",))).stored
    assert embedder.texts == []
    content = "From now on, deploy through staging with PostgreSQL version 16."
    result = mem.capture(f"  {content}  ", kind="constraint", created_at=42,
                         metadata={"namespace": "archic"}, source="hook:claude-code:session")
    assert result.stored and embedder.texts == [content]
    assert result.record.metadata == {"namespace": "archic", "standing": True}
    assert result.record.created_at == 42 and result.record.embedding is not None
    assert not mem.capture(content, kind="constraint", metadata={"namespace": "archic"}).stored
    assert embedder.texts == [content, content]

    novelty = Memory(embedder=embedder, novelty_weight=0.5)
    assert novelty.capture("Production uses PostgreSQL 16.", kind="fact").stored
    assert embedder.texts == [content, content, "Production uses PostgreSQL 16."]


def test_policy_and_instructions_surface_the_parameters() -> None:
    assert "capture" in AGENT_MEMORY_INSTRUCTIONS.lower() and "recall" in AGENT_MEMORY_INSTRUCTIONS.lower()
    summary = policy_summary(DEFAULT_POLICY)
    assert "importance >=" in summary and "duplicate" in summary
