"""Memory stores.

v0 ships an in-memory store with cosine search — vectorised via numpy over a cached embedding matrix
when available, with an identical pure-Python fallback. SQLite (sqlite-vec) and Postgres (pgvector)
are the planned next backends; they implement the same surface so `Memory` is store-agnostic.
"""
from __future__ import annotations

from typing import Callable

from .embeddings import cosine
from .types import MemoryRecord

_VECTORIZE_THRESHOLD = 256  # below this the pure-Python scan is fine (and avoids numpy overhead)
_NUMPY: object = ...  # lazily resolved once; None if numpy is unavailable


def _numpy():
    global _NUMPY
    if _NUMPY is ...:
        try:
            import numpy as _np

            _NUMPY = _np
        except ImportError:
            _NUMPY = None
    return _NUMPY


class InMemoryStore:
    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}
        # Cached embedding matrix for the vectorised scan; rebuilt only when records change so
        # repeated queries on a stable store skip the per-query array build.
        self._matrix = None
        self._matrix_recs: list[MemoryRecord] = []
        self._dirty = True
        # Monotonic change counter: bumps on every mutation. Lets callers cache derived indexes
        # (e.g. the BM25 lexical index for hybrid recall) and invalidate them cheaply.
        self.version = 0

    def put(self, record: MemoryRecord) -> None:
        self._records[record.id] = record
        self._dirty = True
        self.version += 1

    def get(self, record_id: str) -> MemoryRecord | None:
        return self._records.get(record_id)

    def delete(self, record_id: str) -> bool:
        existed = self._records.pop(record_id, None) is not None
        if existed:
            self._dirty = True
            self.version += 1
        return existed

    def all(self) -> list[MemoryRecord]:
        return list(self._records.values())

    def clear(self) -> None:
        self._records.clear()
        self._matrix = None
        self._matrix_recs = []
        self._dirty = True
        self.version += 1

    def _ensure_matrix(self, np) -> None:
        """(Re)build the cached embedding matrix + aligned record list when records have changed."""
        if not self._dirty:
            return
        recs = [r for r in self._records.values() if r.embedding is not None]
        self._matrix_recs = recs
        self._matrix = np.asarray([r.embedding for r in recs], dtype=np.float32) if recs else None
        self._dirty = False

    def search(
        self,
        embedding: list[float],
        *,
        limit: int,
        predicate: Callable[[MemoryRecord], bool] | None = None,
    ) -> list[tuple[float, MemoryRecord]]:
        """Return up to `limit` (cosine, record) pairs, highest similarity first.

        Cosine = dot product (embeddings are L2-normalized). For larger stores this uses a vectorised
        numpy scan over a cached embedding matrix (rebuilt only when records change), with a
        pure-Python fallback that returns identical results. `Memory` stays store-agnostic, so an ANN
        backend can replace this wholesale later."""
        np = _numpy()
        if np is None or len(self._records) <= _VECTORIZE_THRESHOLD:
            records = [
                r
                for r in self._records.values()
                if r.embedding is not None and (predicate is None or predicate(r))
            ]
            if not records:
                return []
            scored = [(cosine(embedding, r.embedding), r) for r in records]
            scored.sort(key=lambda t: t[0], reverse=True)
            return scored[:limit]

        self._ensure_matrix(np)
        if self._matrix is None:
            return []
        recs = self._matrix_recs
        sims = np.clip(self._matrix @ np.asarray(embedding, dtype=np.float32), -1.0, 1.0)

        if predicate is not None:
            mask = np.fromiter((predicate(r) for r in recs), dtype=bool, count=len(recs))
            valid = int(mask.sum())
            if valid == 0:
                return []
            sims = np.where(mask, sims, -2.0)  # filtered-out rows sort below the [-1, 1] range
        else:
            valid = len(recs)

        k = min(limit, valid)
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top], kind="stable")]
        return [(float(sims[i]), recs[i]) for i in top]
