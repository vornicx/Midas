"""Tier-3 optional distillation: Memory(distiller=...) — off by default, fenced, auditable."""
import pytest

from midas import HashingEmbedder, Memory


class FakeDistiller:
    """Deterministic stand-in for a local LLM distiller (no Ollama needed in CI)."""

    def __init__(self, facts):
        self.facts = facts
        self.seen = None

    def distill(self, texts):
        self.seen = list(texts)
        return self.facts


def test_distill_requires_a_configured_distiller():
    mem = Memory(embedder=HashingEmbedder())  # default tier: no distiller
    with pytest.raises(RuntimeError, match="distiller"):
        mem.distill(["some raw turn"])


def test_distill_replace_mode_stores_only_facts():
    # keep_raw=False is the (risky) replace mode — measured catastrophic on BEAM, so opt-in only.
    fake = FakeDistiller(["Deploy target is staging; production deferred."])
    mem = Memory(embedder=HashingEmbedder(), distiller=fake)
    raw = [
        "user: morning!",
        "user: let's just point everything at staging for now, prod isn't ready",
        "user: cool",
    ]
    recs = mem.distill(raw, kind="constraint", keep_raw=False)
    assert fake.seen == raw, "the distiller receives the raw turns"
    assert len(recs) == 1
    assert recs[0].kind == "constraint"
    assert recs[0].metadata["distilled"] is True  # auditable as LLM-derived
    hits = mem.recall("where do we deploy?", limit=3)
    assert hits and "staging" in hits[0].record.content
    assert len(mem.store.all()) == 1


def test_distill_default_keeps_raw_turns():
    # The default is keep_raw=True (measured-safe): facts AUGMENT the verbatim audit trail.
    fake = FakeDistiller(["Deploy target is staging; production deferred."])
    mem = Memory(embedder=HashingEmbedder(), distiller=fake)
    raw = ["user: morning!", "user: point everything at staging, prod isn't ready", "user: cool"]
    mem.distill(raw, kind="constraint")
    contents = [r.content for r in mem.store.all()]
    assert "Deploy target is staging; production deferred." in contents  # the fact
    assert "user: point everything at staging, prod isn't ready" in contents  # raw kept
    assert len(mem.store.all()) == 1 + 3


def test_distill_keep_raw_retains_the_audit_trail():
    fake = FakeDistiller(["The launch date is September 14."])
    mem = Memory(embedder=HashingEmbedder(), distiller=fake)
    raw = ["user: so when do we ship", "user: let's lock September 14"]
    recs = mem.distill(raw, keep_raw=True)
    contents = [r.content for r in mem.store.all()]
    assert "The launch date is September 14." in contents  # distilled fact
    assert "user: let's lock September 14" in contents      # verbatim source kept
    distilled = [r for r in mem.store.all() if r.metadata.get("distilled")]
    raw_kept = [r for r in mem.store.all() if not r.metadata.get("distilled")]
    assert len(distilled) == 1 and len(raw_kept) == 2


def test_default_and_agent_tiers_never_touch_a_distiller():
    # The default tier stores raw; the agent tier stores via capture. Neither needs a distiller.
    mem = Memory(embedder=HashingEmbedder())
    assert mem._distiller is None
    mem.remember("Deploy target is staging.", kind="constraint")  # raw / agent-distilled fact
    mem.capture("My API key rotates on 2027-01-01.", kind="fact")
    assert len(mem.store.all()) == 2
