"""TurboVecStore — a `MemoryStore` backed by a compressed TurboVec index + exact rerank.

The recall lever for long-horizon scale: search a quantized (2/4-bit) `TurboVecIndex` for a cheap
candidate pool (scope allowlist pushed into the kernel), then re-score those candidates by **exact
cosine** on their full-precision vectors and return the top-`limit`. Quantization only generates
candidates; the exact rerank sets the final order — so recall tracks exact search while the *index*
stays ≈8–16× smaller in RAM.

Two modes:
- **default** (no `vector_source`): full embeddings stay in the in-RAM records and feed the rerank.
  Simplest; validates that TurboVec + rerank ≈ exact recall.
- **RAM-saving** (`vector_source=SQLiteVectorSource(...)`): full vectors are written to disk and
  *stripped* from the in-RAM records once indexed, so steady-state RAM is the compressed index alone;
  the rerank reads only the candidate vectors back from disk, and `get()` re-attaches on demand.

The index is maintained **incrementally** — new puts are flushed as a batch before the next search —
so a stripped embedding is never needed to rebuild. Optional extra: `pip install turbovec`.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Collection

from .embeddings import cosine
from .types import MemoryRecord


class TurboVecStore:
    def __init__(
        self,
        *,
        dim: int,
        bit_width: int = 4,
        rerank_pool: int = 200,
        rerank: bool = True,
        vector_source=None,
    ) -> None:
        self.dim = dim
        self._bit_width = bit_width
        self._rerank_pool = rerank_pool
        self._rerank = rerank
        self._vector_source = vector_source  # SQLiteVectorSource | None
        self._records: dict[str, MemoryRecord] = {}
        self._index = None
        self._pending: list[str] = []  # ids put but not yet added to the index
        self.version = 0  # monotonic change counter (lets callers cache derived indexes, e.g. BM25)

    def put(self, record: MemoryRecord) -> None:
        self._records[record.id] = record
        if record.embedding is not None:
            self._pending.append(record.id)
        self.version += 1

    def get(self, record_id: str) -> MemoryRecord | None:
        rec = self._records.get(record_id)
        if rec is not None and rec.embedding is None and self._vector_source is not None:
            vec = self._vector_source.get_many([record_id]).get(record_id)
            if vec is not None:
                rec = replace(rec, embedding=vec)
        return rec

    def delete(self, record_id: str) -> bool:
        existed = self._records.pop(record_id, None) is not None
        if existed:
            if record_id in self._pending:
                self._pending = [p for p in self._pending if p != record_id]
            if self._index is not None:
                self._index.remove(record_id)
            if self._vector_source is not None:
                self._vector_source.remove(record_id)
            self.version += 1
        return existed

    def all(self) -> list[MemoryRecord]:
        return list(self._records.values())

    def clear(self) -> None:
        self._records.clear()
        self._index = None
        self._pending = []
        if self._vector_source is not None:
            self._vector_source.clear()
        self.version += 1

    def _flush(self) -> None:
        """Add buffered puts to the compressed index (incrementally), persist their full vectors to
        the on-disk source, and strip them from RAM. Called before each search."""
        if not self._pending:
            return
        import numpy as np

        from .turbovec_index import TurboVecIndex

        ids = [
            i for i in self._pending
            if self._records.get(i) is not None and self._records[i].embedding is not None
        ]
        self._pending = []
        if not ids:
            return
        if self._index is None:
            self._index = TurboVecIndex(self.dim, bit_width=self._bit_width)
        self._index.add_with_ids(
            np.asarray([self._records[i].embedding for i in ids], dtype=np.float32), ids
        )
        if self._vector_source is not None:
            self._vector_source.add_many(ids, [self._records[i].embedding for i in ids])
            for i in ids:  # free the float32 from RAM — it lives in the index (compressed) + on disk
                self._records[i] = replace(self._records[i], embedding=None)

    def search(
        self,
        embedding: list[float],
        *,
        limit: int,
        predicate: Callable[[MemoryRecord], bool] | None = None,
        allowed_ids: Collection[str] | None = None,
    ) -> list[tuple[float, MemoryRecord]]:
        self._flush()
        if self._index is None:
            return []
        # Resolve the scope to an id-allowlist (pushed in-kernel); AND predicate with allowed_ids.
        if predicate is not None:
            base = set(allowed_ids) if allowed_ids is not None else None
            allow: set[str] | None = {
                r.id for r in self._records.values()
                if predicate(r) and (base is None or r.id in base)
            }
        elif allowed_ids is not None:
            allow = set(allowed_ids)
        else:
            allow = None

        pool = max(self._rerank_pool, limit) if self._rerank else limit
        cand_ids, cand_scores = self._index.search(embedding, k=pool, allowlist=allow)
        cand_ids = [c for c in cand_ids if c]

        if not self._rerank:
            out = [
                (float(s), self._records[c])
                for c, s in zip(cand_ids, cand_scores)
                if c in self._records
            ]
            return out[:limit]

        # Exact cosine rerank on full vectors (from disk in RAM-saving mode, else from RAM).
        full = self._full_vectors(cand_ids)
        scored: list[tuple[float, MemoryRecord]] = []
        for cid in cand_ids:
            rec = self._records.get(cid)
            vec = full.get(cid)
            if rec is None or vec is None:
                continue
            scored.append((cosine(embedding, vec), rec))
        scored.sort(key=lambda t: t[0], reverse=True)
        # Re-attach each returned hit's full vector (cheap — only the top-`limit`), so callers that read
        # `hit.record.embedding` (anchor_boost, novelty) behave the same as with an exact store, while
        # the records kept in the store stay stripped.
        out: list[tuple[float, MemoryRecord]] = []
        for s, rec in scored[:limit]:
            if rec.embedding is None:
                vec = full.get(rec.id)
                if vec is not None:
                    rec = replace(rec, embedding=vec)
            out.append((s, rec))
        return out

    def _full_vectors(self, ids: list[str]) -> dict[str, list[float]]:
        if self._vector_source is not None:
            return self._vector_source.get_many(ids)
        return {
            i: self._records[i].embedding
            for i in ids
            if self._records.get(i) is not None and self._records[i].embedding is not None
        }
