"""Index & store abstractions — the seam vector backends plug into.

Midas keeps its agentic logic (belief revision, forgetting, provenance, context
assembly) independent of *how* vectors are searched. Two small contracts express that:

- `MemoryStore` is the surface `Memory` depends on: record CRUD + a vector `search`
  that returns `(cosine, record)` pairs. `InMemoryStore` (exact scan), `IVFStore`
  (numpy IVF) and `SQLiteStore` (persistent mirror) all satisfy it. A new backend is
  drop-in the moment it implements this — no change to `Memory`.

- `VectorIndex` is the lower, id-mapped index surface a compressed/native backend
  exposes: `add_with_ids` / `remove` / `search(query, k, allowlist)` / `write`.
  TurboVec's `IdMapIndex` matches it almost 1:1 (stable uint64 ids, in-kernel
  allowlist), which is why it slots *under* a `MemoryStore` without disturbing the core.

The **allowlist** is the load-bearing idea. A query's scope (namespace / agent /
project / not-superseded / time window) is a *set of allowed ids*. Pushing that set
into search — O(1) membership here, in-kernel for native backends — beats searching
everything and discarding afterwards: it neither over-fetches nor starves recall.
`MemoryStore.search` takes it as `allowed_ids`; `VectorIndex.search` as `allowlist`.
"""
from __future__ import annotations

from typing import Callable, Collection, Protocol, Sequence, runtime_checkable

from .types import MemoryRecord


@runtime_checkable
class MemoryStore(Protocol):
    """Record store + vector search — the contract `Memory` is written against.

    Embeddings are L2-normalized, so cosine == dot product. `search` returns up to
    `limit` `(cosine, record)` pairs, highest first. `predicate` filters per record;
    `allowed_ids`, when given, restricts results to that id set (AND-ed with
    `predicate`) — the cheap pushdown path for scoped recall. Implementations MAY
    expose a monotonic `version: int` so callers can cache derived indexes (e.g. the
    BM25 lexical index); its absence simply disables that cache.
    """

    def put(self, record: MemoryRecord) -> None: ...
    def get(self, record_id: str) -> MemoryRecord | None: ...
    def delete(self, record_id: str) -> bool: ...
    def all(self) -> list[MemoryRecord]: ...
    def clear(self) -> None: ...
    def search(
        self,
        embedding: list[float],
        *,
        limit: int,
        predicate: Callable[[MemoryRecord], bool] | None = None,
        allowed_ids: Collection[str] | None = None,
    ) -> list[tuple[float, MemoryRecord]]: ...


@runtime_checkable
class VectorIndex(Protocol):
    """Id-mapped ANN index over L2-normalized vectors (cosine).

    The surface a native/compressed backend exposes beneath a `MemoryStore`. Ids are
    stable, caller-owned integers (a backend owns the Midas string-id ↔ uint64 map).
    This mirrors TurboVec's `IdMapIndex`, the Phase-C target:

    - `add_with_ids(vectors, ids)` — online ingest of a batch with explicit ids.
    - `remove(id)` — delete by id (supports forgetting / supersession).
    - `search(query, k, allowlist=...)` — top-k; with `allowlist`, candidates are
      restricted to those ids *inside* the kernel, returning `(ids, scores)`.
    - `write(path)` — persist; restore is the backend's `load(path)` constructor.

    Returned scores may be lossy (quantized vectors); callers that need exact ordering
    rerank the small candidate set against full-precision vectors kept elsewhere —
    Midas already persists those as SQLite BLOBs, so the rerank source is free.
    """

    def add_with_ids(self, vectors: Sequence[Sequence[float]], ids: Sequence[int]) -> None: ...
    def remove(self, id: int) -> bool: ...
    def search(
        self, query: Sequence[float], *, k: int, allowlist: Collection[int] | None = None
    ) -> tuple[Sequence[int], Sequence[float]]: ...
    def write(self, path: str) -> None: ...
