"""Persistent store: SQLite-backed durability + the fast in-memory vectorised search.

Records live in a SQLite file (so they survive restarts) and are mirrored in memory, so search
reuses InMemoryStore's cached numpy cosine scan — **no native extension required** (pure stdlib
sqlite3). The store is safe to share:

  - **across threads** — the connection is guarded by a lock (MCP frameworks run sync tools in
    worker threads, so a per-thread connection assumption breaks in practice);
  - **across processes** — multiple MCP clients (Claude Code + Desktop + an IDE) can point at the
    same DB file. Each process checks SQLite's `PRAGMA data_version` before reads/writes and
    reloads its in-memory mirror when another connection has changed the file, so a memory
    captured in one session is recallable from another *without restarting anything*. WAL mode
    keeps concurrent readers/writers from blocking each other.

For very large shared corpora an ANN backend (sqlite-vec / faiss) behind the same interface
is the next step; this gives persistence + fast search for typical per-agent / per-project memory
today.

    from midas import Memory, LocalEmbedder
    from midas.sqlite_store import SQLiteStore

    mem = Memory(store=SQLiteStore("memory.db"), embedder=LocalEmbedder())
"""
from __future__ import annotations

import json
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Callable

from .store import InMemoryStore
from .types import MemoryRecord

SCHEMA_VERSION = 1  # bump when the `memories` schema changes; add a guarded migration step in _migrate()


class SQLiteStore(InMemoryStore):
    """In-memory store (fast vectorised search) mirrored to a SQLite file for persistence."""

    def __init__(
        self, path: str | Path, *, ann_threshold: int | None = None, ann_nprobe: int = 16
    ) -> None:
        super().__init__(ann_threshold=ann_threshold, ann_nprobe=ann_nprobe)
        self._path = Path(path)
        if self._path.parent and str(self._path.parent):
            self._path.parent.mkdir(parents=True, exist_ok=True)
        # One connection shared across threads, guarded by the lock below: MCP servers execute
        # sync tools in worker threads, so the default same-thread check would crash mid-session.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                kind TEXT NOT NULL,
                importance INTEGER NOT NULL,
                source TEXT,
                provenance TEXT NOT NULL DEFAULT 'observation',
                actor TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                superseded_by TEXT,
                embedding BLOB
            )
            """
        )
        self._migrate()
        self._conn.commit()
        self._load()
        self._data_version = self._current_data_version()

    def _migrate(self) -> None:
        """Bring the schema up to SCHEMA_VERSION (idempotent), and refuse a store written by a NEWER
        Midas so an out-of-date client can't misread it. Runs on every open, so a `midas update` is
        picked up the next time the store opens — there is nothing separate to run."""
        cur = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if cur > SCHEMA_VERSION:
            raise RuntimeError(
                f"This memory store uses schema v{cur}, newer than this Midas (v{SCHEMA_VERSION}). "
                f"Upgrade Midas first:  midas update"
            )
        # Legacy/uninitialized stores: ensure the columns exist (older files predate these).
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(memories)").fetchall()}
        if "provenance" not in columns:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN provenance TEXT NOT NULL DEFAULT 'observation'"
            )
        if "actor" not in columns:
            self._conn.execute("ALTER TABLE memories ADD COLUMN actor TEXT")
        # --- future migrations go here: `if cur < 2: <ALTER/UPDATE…>` then bump SCHEMA_VERSION ---
        if cur != SCHEMA_VERSION:
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def schema_version(self) -> int:
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def _load(self) -> None:
        cur = self._conn.execute(
            "SELECT id, content, kind, importance, source, provenance, actor, metadata_json, "
            "created_at, updated_at, superseded_by, embedding FROM memories"
        )
        for row in cur.fetchall():
            super().put(self._row_to_record(row))  # in-memory only; already on disk

    def _current_data_version(self) -> int:
        # `PRAGMA data_version` changes when the database is modified by a DIFFERENT connection
        # (another process, or another connection in this one) — never by our own writes. That
        # makes it a cheap staleness probe: no change means the in-memory mirror is current.
        return int(self._conn.execute("PRAGMA data_version").fetchone()[0])

    def _refresh_if_stale(self) -> None:
        """Reload the in-memory mirror when another connection has written the DB file.

        This is what lets several agent processes share one memory file live: each sees the
        others' captures on its next read instead of holding a frozen snapshot from startup.
        """
        version = self._current_data_version()
        if version == self._data_version:
            return
        self._data_version = version
        InMemoryStore.clear(self)  # mirror only — the rows on disk are the source of truth
        self._load()

    @staticmethod
    def _row_to_record(row) -> MemoryRecord:
        (id_, content, kind, importance, source, provenance, actor, metadata_json,
         created_at, updated_at, superseded_by, emb_blob) = row
        embedding = None
        if emb_blob is not None:
            try:
                import numpy as np

                embedding = np.frombuffer(emb_blob, dtype="<f4").copy()  # float32 array (footprint)
            except ImportError:
                embedding = list(struct.unpack(f"<{len(emb_blob) // 4}f", emb_blob))
        return MemoryRecord(
            id=id_, content=content, kind=kind, importance=importance, source=source,
            provenance=provenance or "observation", actor=actor,
            metadata=json.loads(metadata_json) if metadata_json else {},
            created_at=created_at, updated_at=updated_at,
            superseded_by=superseded_by, embedding=embedding,
        )

    def put(self, record: MemoryRecord) -> None:
        with self._lock:
            self._refresh_if_stale()
            super().put(record)  # in-memory + marks the search cache dirty
            emb_blob = (
                struct.pack(f"<{len(record.embedding)}f", *record.embedding)
                if record.embedding is not None
                else None
            )
            self._conn.execute(
                """
                INSERT INTO memories (id, content, kind, importance, source, provenance, actor,
                                      metadata_json,
                                      created_at, updated_at, superseded_by, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content=excluded.content, kind=excluded.kind, importance=excluded.importance,
                    source=excluded.source, provenance=excluded.provenance, actor=excluded.actor,
                    metadata_json=excluded.metadata_json,
                    created_at=excluded.created_at, updated_at=excluded.updated_at,
                    superseded_by=excluded.superseded_by, embedding=excluded.embedding
                """,
                (record.id, record.content, record.kind, record.importance, record.source,
                 record.provenance, record.actor, json.dumps(record.metadata or {}),
                 record.created_at, record.updated_at, record.superseded_by, emb_blob),
            )
            self._conn.commit()

    def get(self, record_id: str) -> MemoryRecord | None:
        with self._lock:
            self._refresh_if_stale()
            return super().get(record_id)

    def all(self) -> list[MemoryRecord]:
        with self._lock:
            self._refresh_if_stale()
            return super().all()

    def search(
        self,
        embedding: list[float],
        *,
        limit: int,
        predicate: Callable[[MemoryRecord], bool] | None = None,
    ) -> list[tuple[float, MemoryRecord]]:
        with self._lock:
            self._refresh_if_stale()
            return super().search(embedding, limit=limit, predicate=predicate)

    def delete(self, record_id: str) -> bool:
        with self._lock:
            self._refresh_if_stale()
            existed = super().delete(record_id)
            if existed:
                self._conn.execute("DELETE FROM memories WHERE id = ?", (record_id,))
                self._conn.commit()
            return existed

    def clear(self) -> None:
        with self._lock:
            super().clear()
            self._conn.execute("DELETE FROM memories")
            self._conn.commit()

    def close(self) -> None:
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None
