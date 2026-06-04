"""Approximate nearest-neighbour search for scaling Midas past the exact in-memory scan.

`InMemoryStore`'s cached scan is exact but O(N) per query (~230 ms at 1M x 768-d). `IVFIndex` is an
**inverted-file** index built with **numpy only** (no native dependency, unlike faiss/hnswlib): it
clusters the corpus into `nlist` cells via k-means, and a query then compares against only the
`nprobe` nearest cells instead of the whole corpus. That makes search **sub-linear** -- at 1M with
`nlist=1000, nprobe=8` a query scans ~8K vectors instead of 1M (~100x fewer) -- trading a little
recall for a large speedup. `nprobe` tunes the recall/latency trade-off at query time (no rebuild).

`IVFStore` wraps the index behind the same store surface as `InMemoryStore`, so `Memory` can use it
unchanged. The index is built lazily and rebuilt on mutation, so it suits **read-heavy / batch-loaded**
corpora (build once, query many); for write-heavy workloads keep `InMemoryStore`.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .types import MemoryRecord


def _kmeans(x: np.ndarray, k: int, *, n_iter: int, rng: np.random.Generator) -> np.ndarray:
    """Spherical k-means (cosine): centroids are L2-normalized, assignment is by max dot product.

    Inputs are assumed L2-normalized (Midas embeddings are), so dot product == cosine similarity.
    """
    # Init from k distinct points.
    centroids = x[rng.choice(x.shape[0], k, replace=False)].copy()
    for _ in range(n_iter):
        assign = np.argmax(x @ centroids.T, axis=1)
        for c in range(k):
            members = x[assign == c]
            if len(members):
                v = members.sum(axis=0)
                norm = float(np.linalg.norm(v))
                if norm > 0.0:
                    centroids[c] = v / norm
            else:  # empty cell -> reseed from a random point so it can pick up members next round
                centroids[c] = x[rng.integers(x.shape[0])]
    return centroids


class IVFIndex:
    """Inverted-file ANN index over L2-normalized vectors (cosine similarity)."""

    def __init__(self, nlist: int | None = None, *, n_iter: int = 10, train_sample: int = 50_000,
                 seed: int = 0) -> None:
        self.nlist = nlist
        self.n_iter = n_iter
        self.train_sample = train_sample
        self.seed = seed
        self._centroids: np.ndarray | None = None
        self._lists: list[np.ndarray] = []      # cell -> row indices into _vectors
        self._vectors: np.ndarray | None = None  # (N, d) float32, L2-normalized

    def fit(self, vectors: Sequence[Sequence[float]] | np.ndarray) -> "IVFIndex":
        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("fit() needs a non-empty (N, d) matrix")
        n = x.shape[0]
        nlist = self.nlist or max(1, int(round(np.sqrt(n))))
        nlist = min(nlist, n)
        self.nlist = nlist
        rng = np.random.default_rng(self.seed)
        # Train centroids on a sample (k-means cost is O(sample x nlist x d x iters)); assign all.
        train = x if n <= self.train_sample else x[rng.choice(n, self.train_sample, replace=False)]
        centroids = _kmeans(train, nlist, n_iter=self.n_iter, rng=rng)
        assign = np.argmax(x @ centroids.T, axis=1)
        self._centroids = centroids
        self._lists = [np.where(assign == c)[0] for c in range(nlist)]
        self._vectors = x
        return self

    def search(self, query: Sequence[float], *, k: int = 10, nprobe: int = 8,
               allowed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (row indices, scores) for the top-k, highest cosine first.

        `allowed` is an optional boolean mask over rows (predicate pushdown); candidates failing it
        are dropped before the top-k cut.
        """
        if self._vectors is None or self._centroids is None:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        q = np.asarray(query, dtype=np.float32)
        nprobe = max(1, min(nprobe, self.nlist))
        csims = self._centroids @ q
        cells = np.argpartition(-csims, nprobe - 1)[:nprobe]
        candidates = np.concatenate([self._lists[c] for c in cells]) if len(cells) else np.empty(0, int)
        if candidates.size and allowed is not None:
            candidates = candidates[allowed[candidates]]
        if candidates.size == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        sims = self._vectors[candidates] @ q
        kk = min(k, candidates.size)
        part = np.argpartition(-sims, kk - 1)[:kk]
        order = part[np.argsort(-sims[part], kind="stable")]
        return candidates[order], sims[order].astype(np.float32)


class IVFStore:
    """Store with the `InMemoryStore` surface, backed by an `IVFIndex` for sub-linear search.

    Build is lazy and triggered on the first search after a mutation. `nprobe` (recall/latency knob)
    is set at construction. Best for read-heavy corpora; for write-heavy use `InMemoryStore`.
    """

    def __init__(self, *, nlist: int | None = None, nprobe: int = 8, seed: int = 0) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._nlist = nlist
        self.nprobe = nprobe
        self.seed = seed
        self._index: IVFIndex | None = None
        self._rows: list[MemoryRecord] = []  # row -> record (aligned with the fitted matrix)
        self._dirty = True

    def put(self, record: MemoryRecord) -> None:
        self._records[record.id] = record
        self._dirty = True

    def get(self, record_id: str) -> MemoryRecord | None:
        return self._records.get(record_id)

    def delete(self, record_id: str) -> bool:
        existed = self._records.pop(record_id, None) is not None
        self._dirty = self._dirty or existed
        return existed

    def all(self) -> list[MemoryRecord]:
        return list(self._records.values())

    def clear(self) -> None:
        self._records.clear()
        self._index = None
        self._rows = []
        self._dirty = True

    def _ensure_index(self) -> None:
        if not self._dirty:
            return
        recs = [r for r in self._records.values() if r.embedding is not None]
        self._rows = recs
        self._index = IVFIndex(nlist=self._nlist, seed=self.seed).fit(
            [r.embedding for r in recs]) if recs else None
        self._dirty = False

    def search(self, embedding: list[float], *, limit: int,
               predicate: Callable[[MemoryRecord], bool] | None = None
               ) -> list[tuple[float, MemoryRecord]]:
        self._ensure_index()
        if self._index is None:
            return []
        allowed = None
        if predicate is not None:
            allowed = np.fromiter((predicate(r) for r in self._rows), dtype=bool, count=len(self._rows))
        rows, scores = self._index.search(embedding, k=limit, nprobe=self.nprobe, allowed=allowed)
        return [(float(s), self._rows[i]) for i, s in zip(rows, scores)]
