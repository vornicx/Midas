"""TurboVec-backed VectorIndex — compressed (2/4-bit) ANN with an in-kernel allowlist.

Wraps TurboVec's `IdMapIndex` (TurboQuant quantization + SIMD search) behind Midas' `VectorIndex`
contract. TurboVec keys on uint64 ids; Midas records are string-id'd, so this adapter owns the
stable `str ↔ uint64` map. Vectors are L2-normalized float32 (Midas embeddings already are); `search`
takes one query and returns `(ids, scores)` with the candidates' Midas string ids, best-first.

TurboVec keeps only the *compressed* vectors (≈8–16× less RAM at 4/2-bit) and discards the originals,
so callers that need exact ordering rerank the returned candidates against full-precision vectors
kept elsewhere — Midas already persists those as SQLite BLOBs, so the rerank source is free.

Optional extra: `pip install turbovec` (prebuilt wheel, no torch). Build is `add_with_ids` →
`search`; the index is `prepare()`d lazily on the first search after a mutation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Collection, Sequence


class TurboVecIndex:
    """`VectorIndex` over TurboVec's `IdMapIndex`, owning the Midas-string-id ↔ uint64 map."""

    def __init__(self, dim: int, *, bit_width: int = 4) -> None:
        import turbovec  # lazy, optional dependency

        self.dim = dim
        self.bit_width = bit_width
        self._idx = turbovec.IdMapIndex(dim=dim, bit_width=bit_width)
        self._to_uint: dict[str, int] = {}
        self._to_str: dict[int, str] = {}
        self._next = 1  # 0 reserved as "missing"
        self._needs_prepare = False

    def _uint_for(self, sid: str) -> int:
        u = self._to_uint.get(sid)
        if u is None:
            u = self._next
            self._next += 1
            self._to_uint[sid] = u
            self._to_str[u] = sid
        return u

    def add_with_ids(self, vectors: Sequence[Sequence[float]], ids: Sequence[str]) -> None:
        import numpy as np

        ids = list(ids)
        uids = np.array([self._uint_for(s) for s in ids], dtype=np.uint64)
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.dim or vecs.shape[0] != len(ids):
            raise ValueError(f"expected ({len(ids)}, {self.dim}) vectors, got {vecs.shape}")
        self._idx.add_with_ids(vecs, uids)
        self._needs_prepare = True

    def remove(self, id: str) -> bool:
        u = self._to_uint.pop(id, None)
        if u is None:
            return False
        self._to_str.pop(u, None)
        try:
            self._idx.remove(u)
        except Exception:
            pass
        return True

    def search(
        self, query: Sequence[float], *, k: int, allowlist: Collection[str] | None = None
    ) -> tuple[list[str], list[float]]:
        import numpy as np

        if self._needs_prepare:
            self._idx.prepare()
            self._needs_prepare = False
        q = np.ascontiguousarray(np.asarray(query, dtype=np.float32).reshape(1, -1))
        if allowlist is not None:
            allow = np.array(
                [self._to_uint[s] for s in allowlist if s in self._to_uint], dtype=np.uint64
            )
            scores, uids = self._idx.search(q, k, allowlist=allow)
        else:
            scores, uids = self._idx.search(q, k)
        ids = [self._to_str.get(int(u), "") for u in uids[0]]
        return ids, [float(s) for s in scores[0]]

    def __len__(self) -> int:
        return len(self._to_uint)

    def write(self, path: str) -> None:
        self._idx.write(path)
        # TurboVec persists only the compressed vectors + uint64 ids — store the string-id map next to it.
        Path(f"{path}.ids.json").write_text(
            json.dumps({"to_str": {str(u): s for u, s in self._to_str.items()}, "next": self._next})
        )

    @classmethod
    def load(cls, path: str) -> "TurboVecIndex":
        import turbovec

        inner = turbovec.IdMapIndex.load(path)
        obj = cls.__new__(cls)
        obj._idx = inner
        obj.dim = inner.dim
        obj.bit_width = inner.bit_width
        meta = json.loads(Path(f"{path}.ids.json").read_text())
        obj._to_str = {int(u): s for u, s in meta["to_str"].items()}
        obj._to_uint = {s: u for u, s in obj._to_str.items()}
        obj._next = meta["next"]
        obj._needs_prepare = False
        return obj
