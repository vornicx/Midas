"""On-disk id → embedding store (SQLite BLOB) — the full-precision side of compressed search.

A compressed index (TurboVec, 2/4-bit) keeps only quantized vectors in RAM; the full-precision
vectors live here on disk and are read back **only for the small candidate set** being reranked.
That is what turns TurboVec's index compression into an end-to-end RAM win: the float32 matrix never
sits in memory, just a few hundred vectors per query during the exact rerank.

Pure stdlib `sqlite3` + `struct` (no native extension, no numpy required); vectors are stored as
little-endian float32, the same encoding `SQLiteStore` uses for its `embedding` BLOBs.
"""
from __future__ import annotations

import sqlite3
import struct
import threading
from pathlib import Path
from typing import Iterable, Sequence

_CHUNK = 900  # stay under SQLite's default 999-variable limit per statement


class SQLiteVectorSource:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if str(self._path.parent):
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors (id TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL)"
        )
        self._conn.commit()

    def add_many(self, ids: Sequence[str], vectors: Iterable[Sequence[float]]) -> None:
        rows = []
        for rid, vec in zip(ids, vectors):
            v = [float(x) for x in vec]
            rows.append((rid, len(v), struct.pack(f"<{len(v)}f", *v)))
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT INTO vectors (id, dim, vec) VALUES (?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET dim=excluded.dim, vec=excluded.vec",
                rows,
            )
            self._conn.commit()

    def get_many(self, ids: Sequence[str]) -> dict[str, list[float]]:
        ids = list(ids)
        if not ids:
            return {}
        out: dict[str, list[float]] = {}
        with self._lock:
            for i in range(0, len(ids), _CHUNK):
                chunk = ids[i : i + _CHUNK]
                placeholders = ",".join("?" * len(chunk))
                cur = self._conn.execute(
                    f"SELECT id, dim, vec FROM vectors WHERE id IN ({placeholders})", chunk
                )
                for rid, dim, blob in cur.fetchall():
                    out[rid] = list(struct.unpack(f"<{dim}f", blob))
        return out

    def remove(self, id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM vectors WHERE id = ?", (id,))
            self._conn.commit()
            return cur.rowcount > 0

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM vectors")
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])

    def close(self) -> None:
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None
