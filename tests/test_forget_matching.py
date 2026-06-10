"""Topic-level erasure: forget_matching (dry-run audit + deletion) and chain-safe forget()."""
from midas import HashingEmbedder, Memory


def _mem(**kwargs) -> Memory:
    return Memory(embedder=HashingEmbedder(), **kwargs)


def test_forget_matching_dry_run_deletes_nothing():
    mem = _mem()
    mem.remember("my home address is 12 Elm Street", kind="fact", importance=5)
    mem.remember("the build uses Python 3.12", kind="fact", importance=3)

    preview = mem.forget_matching("home address", min_relevance=0.3, dry_run=True)
    assert preview and any("Elm Street" in r.content for r in preview)
    assert len(mem.store.all()) == 2, "dry run must not delete"


def test_forget_matching_deletes_only_matches_and_returns_audit():
    mem = _mem()
    mem.remember("my home address is 12 Elm Street", kind="fact", importance=5)
    mem.remember("the build uses Python 3.12", kind="fact", importance=3)

    audit = mem.forget_matching("home address", min_relevance=0.3)
    assert any("Elm Street" in r.content for r in audit)
    remaining = [r.content for r in mem.store.all()]
    assert remaining == ["the build uses Python 3.12"]


def test_forget_matching_bypasses_durability_protection():
    # Explicit erasure outranks retention: even a protected, max-importance constraint goes.
    mem = _mem()
    mem.remember("the user's SSN ends in 6789", kind="constraint", importance=5)
    audit = mem.forget_matching("SSN", min_relevance=0.1)
    assert audit and mem.store.all() == []


def test_forget_repairs_supersession_chain():
    mem = _mem(supersede=True, supersede_threshold=0.3, supersede_lower_threshold=0.1)
    old = mem.remember("the launch date is September 14", kind="fact", importance=4)
    mid = mem.remember("actually the launch date moved to October 2", kind="fact", importance=4)
    new = mem.remember("actually the launch date moved again to November 5", kind="fact", importance=4)
    assert mem.store.get(old.id).superseded_by is not None, "chain should exist"

    # Delete the middle of the chain; the old record must relink, not dangle.
    assert mem.forget(mid.id) is True
    head = mem._resolve_head(mem.store.get(old.id))
    assert head.id == new.id, "resolve_head must still reach the current belief"


def test_forget_missing_id_returns_false():
    assert _mem().forget("nope") is False
