"""SDK metadata_filter: equality scoping for recall/build_context (namespaces, users, projects)."""
from midas import HashingEmbedder, Memory


def _mem() -> Memory:
    return Memory(embedder=HashingEmbedder())


def test_recall_scopes_by_metadata_filter():
    mem = _mem()
    mem.remember("the api gateway is Kong", kind="fact", metadata={"namespace": "proj-a"})
    mem.remember("the api gateway is Envoy", kind="fact", metadata={"namespace": "proj-b"})

    a = mem.recall("api gateway", metadata_filter={"namespace": "proj-a"})
    assert a and all(h.record.metadata["namespace"] == "proj-a" for h in a)
    assert any("Kong" in h.record.content for h in a)
    assert not any("Envoy" in h.record.content for h in a)

    unscoped = mem.recall("api gateway", limit=5)
    assert {h.record.metadata["namespace"] for h in unscoped} == {"proj-a", "proj-b"}


def test_metadata_filter_requires_all_keys():
    mem = _mem()
    mem.remember("note one", metadata={"namespace": "a", "user": "u1"})
    mem.remember("note two", metadata={"namespace": "a", "user": "u2"})
    hits = mem.recall("note", metadata_filter={"namespace": "a", "user": "u2"}, limit=5)
    assert len(hits) == 1 and hits[0].record.content == "note two"


def test_build_context_passes_metadata_filter_through():
    mem = _mem()
    mem.remember("launch is on March 3", kind="fact", metadata={"namespace": "a"})
    mem.remember("launch is on July 9", kind="fact", metadata={"namespace": "b"})
    text = mem.assemble("when is the launch?", metadata_filter={"namespace": "b"})
    assert "July 9" in text and "March 3" not in text


def test_hybrid_recall_respects_metadata_filter():
    mem = _mem()
    mem.remember("ticket ABC-123 tracks the login bug", kind="note", metadata={"namespace": "a"})
    mem.remember("ticket ABC-123 duplicated in tracker", kind="note", metadata={"namespace": "b"})
    hits = mem.recall("ABC-123", hybrid=True, metadata_filter={"namespace": "a"}, limit=5)
    assert hits and all(h.record.metadata["namespace"] == "a" for h in hits)
