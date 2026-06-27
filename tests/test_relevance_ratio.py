"""Scale-free parsimony: the min_relevance_ratio floor (default 0.3, measured-safe)."""
from midas import HashingEmbedder, Memory


def _seed(mem: Memory) -> None:
    mem.remember("the launch date is September 14", kind="fact", importance=3)
    mem.remember("completely unrelated quarterly parking rota", kind="note", importance=3)


def test_default_ratio_prunes_far_below_top_hit():
    mem = Memory(embedder=HashingEmbedder())
    _seed(mem)
    hits = mem.recall("when is the launch date?", limit=5)
    assert hits and "September 14" in hits[0].record.content
    assert all(
        h.relevance >= 0.3 * hits[0].relevance for h in hits
    ), "nothing far below the top hit should survive"


def test_ratio_zero_disables_pruning():
    mem = Memory(embedder=HashingEmbedder(), min_relevance_ratio=0)
    _seed(mem)
    # ratio=0 disables the RELATIVE floor; min_relevance=0 also disables the embedder's absolute noise
    # floor, so the unrelated (≈0-relevance) record survives too.
    assert len(mem.recall("when is the launch date?", limit=5, min_relevance=0)) == 2


def test_per_call_ratio_overrides_instance_default():
    mem = Memory(embedder=HashingEmbedder())
    _seed(mem)
    assert len(mem.recall("when is the launch date?", limit=5, min_relevance_ratio=0, min_relevance=0)) == 2
    strict = mem.recall("when is the launch date?", limit=5, min_relevance_ratio=0.99)
    assert len(strict) == 1 and "September 14" in strict[0].record.content


def test_hashing_noise_floor_drops_the_lone_irrelevant_hit():
    """The reported issue: with the hashing fallback, a query that matches nothing used to return the
    only memory anyway. The embedder's absolute floor now returns empty instead — without dropping a
    real match (which lands well above the floor)."""
    mem = Memory(embedder=HashingEmbedder())
    mem.remember("User prefers dark mode.", kind="preference")
    assert mem.recall("the home mailing address") == []   # no token overlap → nothing, not junk
    assert mem.recall("dark mode preference")              # a real match still surfaces
