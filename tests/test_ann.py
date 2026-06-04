"""IVF ANN index + store: recall vs exact, drop-in store behaviour, predicate pushdown."""
import numpy as np
import pytest

from midas.ann import IVFIndex, IVFStore
from midas.store import InMemoryStore
from midas.types import MemoryRecord


def _clustered(n_per: int, k: int, d: int, *, spread: float = 0.05, seed: int = 0) -> np.ndarray:
    """k well-separated clusters of unit vectors — the structure real embeddings have."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    pts = np.vstack([c + spread * rng.normal(size=(n_per, d)) for c in centers])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts.astype(np.float32)


def _exact_topk(x: np.ndarray, q: np.ndarray, k: int) -> set[int]:
    return set(np.argsort(-(x @ q))[:k].tolist())


def test_ivf_matches_exact_when_probing_all_cells():
    x = _clustered(50, 8, 32)
    idx = IVFIndex(nlist=8, seed=0).fit(x)
    q = x[7]
    got, _ = idx.search(q, k=10, nprobe=8)  # nprobe == nlist => exhaustive => exact
    assert set(got.tolist()) == _exact_topk(x, q, 10)


def test_ivf_high_recall_at_modest_nprobe():
    x = _clustered(100, 16, 64)
    idx = IVFIndex(seed=0).fit(x)  # nlist ~= sqrt(1600) = 40
    rng = np.random.default_rng(1)
    qs = x[rng.choice(len(x), 50, replace=False)]
    hits = sum(len(_exact_topk(x, q, 10) & set(idx.search(q, k=10, nprobe=8)[0].tolist())) for q in qs)
    assert hits / (len(qs) * 10) >= 0.8  # clustered data => high recall even probing few cells


def test_ivf_scores_descending():
    x = _clustered(40, 5, 32)
    idx = IVFIndex(nlist=5, seed=0).fit(x)
    _, scores = idx.search(x[0], k=10, nprobe=5)
    assert np.all(np.diff(scores) <= 1e-6)


def test_ivfstore_dropin_matches_inmemory():
    x = _clustered(40, 4, 32)
    recs = [MemoryRecord(id=str(i), content=f"r{i}", embedding=x[i].tolist()) for i in range(len(x))]
    mem, ivf = InMemoryStore(), IVFStore(nlist=4, nprobe=4, seed=0)  # nprobe==nlist => exact
    for r in recs:
        mem.put(r)
        ivf.put(r)
    q = x[3].tolist()
    assert ivf.search(q, limit=1)[0][1].id == mem.search(q, limit=1)[0][1].id == "3"


def test_ivfstore_predicate_and_empty():
    ivf = IVFStore(nlist=4, nprobe=4, seed=0)
    assert ivf.search([0.0] * 16, limit=5) == []  # empty store
    x = _clustered(20, 4, 16)
    for i in range(len(x)):
        ivf.put(MemoryRecord(id=str(i), content="x", embedding=x[i].tolist(),
                             kind="fact" if i % 2 else "note"))
    res = ivf.search(x[1].tolist(), limit=50, predicate=lambda r: r.kind == "fact")
    assert res and all(r.kind == "fact" for _, r in res)


def test_ivf_empty_fit_raises():
    with pytest.raises(ValueError):
        IVFIndex().fit(np.empty((0, 8), dtype=np.float32))
