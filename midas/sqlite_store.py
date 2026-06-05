"""Persistent store: SQLite-backed durability + the fast in-memory vectorised search.

Records live in a SQLite file (so they survive restarts) and are mirrored in memory, so search
reuses InMemoryStore's cached numpy cosine scan — **no native extension required** (pure stdlib
sqlite3). For very large shared corpora an ANN backend (sqlite-vec / faiss) behind the same interface
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
from pathlib import Path

from .store import InMemoryStore
from .types import MemoryRecord


class SQLiteStore(InMemoryStore):
    """In-memory store (fast vectorised search) mirrored to a SQLite file for persistence."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self._path = Path(path)
        if self._path.parent and str(self._path.parent):
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                kind TEXT NOT NULL,
                importance INTEGER NOT NULL,
                source TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                superseded_by TEXT,
                embedding BLOB
            )
            """
        )
        self._conn.commit()
        self._load()

    def _load(self) -> None:
        cur = self._conn.execute(
            "SELECT id, content, kind, importance, source, metadata_json, "
            "created_at, updated_at, superseded_by, embedding FROM memories"
        )
        for row in cur.fetchall():
            super().put(self._row_to_record(row))  # in-memory only; already on disk

    @staticmethod
    def _row_to_record(row) -> MemoryRecord:
        (id_, content, kind, importance, source, metadata_json,
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
            metadata=json.loads(metadata_json) if metadata_json else {},
            created_at=created_at, updated_at=updated_at,
            superseded_by=superseded_by, embedding=embedding,
        )

    def put(self, record: MemoryRecord) -> None:
        super().put(record)  # in-memory + marks the search cache dirty
        emb_blob = (
            struct.pack(f"<{len(record.embedding)}f", *record.embedding)
            if record.embedding is not None
            else None
        )
        self._conn.execute(
            """
            INSERT INTO memories (id, content, kind, importance, source, metadata_json,
                                  created_at, updated_at, superseded_by, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                content=excluded.content, kind=excluded.kind, importance=excluded.importance,
                source=excluded.source, metadata_json=excluded.metadata_json,
                created_at=excluded.created_at, updated_at=excluded.updated_at,
                superseded_by=excluded.superseded_by, embedding=excluded.embedding
            """,
            (record.id, record.content, record.kind, record.importance, record.source,
             json.dumps(record.metadata or {}), record.created_at, record.updated_at,
             record.superseded_by, emb_blob),
        )
        self._conn.commit()

    def delete(self, record_id: str) -> bool:
        existed = super().delete(record_id)
        if existed:
            self._conn.execute("DELETE FROM memories WHERE id = ?", (record_id,))
            self._conn.commit()
        return existed

    def clear(self) -> None:
        super().clear()
        self._conn.execute("DELETE FROM memories")
        self._conn.commit()

    def close(self) -> None:
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None
