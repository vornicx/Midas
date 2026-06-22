"""allowed_ids — the id-allowlist seam for predicate pushdown (Phase B).

The allowlist restricts a vector search to a caller-supplied id set, evaluated as
O(1) membership instead of a per-row Python predicate. It is what id-based backends
filter against (IVF's row-mask today, TurboVec's in-kernel uint64 allowlist next),
and it must be exactly equivalent to filtering by the matching predicate.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas import InMemoryStore, MemoryRecord, MemoryStore
from midas.ann import IVFStore


def _rec(i: int, vec) -> MemoryRecord:
    v = np.asarray(vec, dtype=np.float32)
    v = v / (float(np.linalg.norm(v)) or 1.0)
    return MemoryRecord(id=f"r{i}", content=f"c{i}", embedding=[float(x) for x in v])


def _make(store):
    # Two clusters on a ring so cosine ordering is well-defined: r0..r2 near (1,0), r3..r5 near (0,1).
    for i, v in enumerate([(1, 0), (0.9, 0.1), (0.8, 0.2), (0.2, 0.8), (0.1, 0.9), (0, 1)]):
        store.put(_rec(i, v))
    return store


_BACKENDS = [InMemoryStore, lambda: IVFStore(nlist=2, nprobe=2)]


@pytest.mark.parametrize("make", _BACKENDS)
def test_allowed_ids_is_a_hard_filter(make):
    store = _make(make())
    hits = store.search([1.0, 0.0], limit=10, allowed_ids={"r3", "r4", "r5"})
    assert {r.id for _, r in hits} == {"r3", "r4", "r5"}  # nothing outside the allowlist, even if closer


@pytest.mark.parametrize("make", _BACKENDS)
def test_allowed_ids_none_is_unfiltered(make):
    store = _make(make())
    base = {r.id for _, r in store.search([1.0, 0.0], limit=10)}
    again = {r.id for _, r in store.search([1.0, 0.0], limit=10, allowed_ids=None)}
    assert base == again == {f"r{i}" for i in range(6)}


def test_allowed_ids_intersects_predicate():
    store = _make(InMemoryStore())
    hits = store.search(
        [1.0, 0.0], limit=10, predicate=lambda r: r.id != "r4", allowed_ids={"r3", "r4", "r5"}
    )
    assert {r.id for _, r in hits} == {"r3", "r5"}  # r4 in the allowlist but dropped by the predicate


def test_empty_allowlist_returns_nothing():
    store = _make(InMemoryStore())
    assert store.search([1.0, 0.0], limit=10, allowed_ids=set()) == []


def test_store_conforms_to_protocol():
    assert isinstance(InMemoryStore(), MemoryStore)
