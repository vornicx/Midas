"""SQLite persistence with bounded exact vector search and a tamper-evident audit log.

SQLite holds the authoritative records and vectors. Exact search reads batches from one
SQLite snapshot and retains just the best candidates. Explicit IVF mode keeps an in-memory
mirror for its approximate index. WAL makes other clients' committed writes visible.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import os
import sqlite3
import struct
import threading
import time
from pathlib import Path
from collections.abc import Callable, Collection, Iterator
from contextlib import nullcontext

from .embeddings import cosine
from .store import InMemoryStore, _numpy
from .types import MemoryRecord

SCHEMA_VERSION = 1  # bump when the `memories` schema changes; add a guarded migration step in _migrate()

# The audit chain's genesis value — the prev_hash of the first entry.
_AUDIT_GENESIS = "0" * 64
_COLUMNS = ("id, content, kind, importance, source, provenance, actor, metadata_json, "
            "created_at, updated_at, superseded_by")
_SEARCH_BATCH = 256
_SQL_VARIABLE_LIMIT = 900
_NAMESPACE_SQL = "CASE WHEN json_valid(metadata_json) THEN json_extract(metadata_json, '$.namespace') END"


def _audit_hash(seq: int, at: float, op: str, record_id: str, content_sha: str,
                prev_hash: str) -> str:
    # `at` is canonicalised to integer microseconds (floor) — float FORMATTING differs across
    # runtimes, but IEEE multiply+floor of the same stored double doesn't, so the TypeScript port
    # verifies chains written by Python and vice versa.
    return hashlib.sha256(
        f"{seq}|{int(at * 1_000_000)}|{op}|{record_id}|{content_sha}|{prev_hash}".encode()
    ).hexdigest()


def _content_sha(record: MemoryRecord) -> str:
    """Fingerprint of the record's audited state: id + content + revision link. Enough to later prove
    what a mutation touched; NOT enough to reconstruct the content (the audit log stores no text)."""
    payload = f"{record.id}\x00{record.content}\x00{record.superseded_by or ''}"
    return hashlib.sha256(payload.encode()).hexdigest()


class SQLiteStore(InMemoryStore):
    """Exact on-disk search by default, with an opt-in in-memory IVF index."""

    def __init__(
        self, path: str | Path, *, ann_threshold: int | None = None, ann_nprobe: int = 16,
        audit: bool = True, key: str | None = None,
    ) -> None:
        super().__init__(ann_threshold=ann_threshold, ann_nprobe=ann_nprobe)
        self._audit_enabled = audit
        self._path = Path(path)
        if self._path.parent and str(self._path.parent):
            self._path.parent.mkdir(parents=True, exist_ok=True)
        # One connection shared across threads, guarded by the lock below: MCP servers execute
        # sync tools in worker threads, so the default same-thread check would crash mid-session.
        self._lock = threading.RLock()
        # Encryption at rest (opt-in): with `key` (or MIDAS_MCP_KEY), the file is a SQLCipher
        # database — unreadable without the key. FAILS CLOSED: if the user asked for encryption and
        # SQLCipher isn't installed, refuse rather than silently writing plaintext.
        key = key if key is not None else (os.getenv("MIDAS_MCP_KEY") or None)
        if key:
            try:
                from sqlcipher3 import dbapi2 as _sqlcipher
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise RuntimeError(
                    'MIDAS_MCP_KEY is set but SQLCipher is not installed — '
                    'pip install "midas-memory[encrypted]"'
                ) from exc
            self._conn = _sqlcipher.connect(str(self._path), check_same_thread=False)
            self._conn.execute("PRAGMA key = '{}'".format(key.replace("'", "''")))
        else:
            self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        try:
            self._conn.execute("SELECT json_valid('{}'), json_extract('{}', '$.namespace')").fetchone()
            self._sql_namespaces = True
        except self._conn.OperationalError:  # Older SQLite/SQLCipher builds can lack JSON1.
            self._sql_namespaces = False
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
        # Tamper-evident mutation log (append-only, hash-chained; additive table so older Midas
        # versions can still open the file). Rows carry hashes, never memory content.
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                seq INTEGER PRIMARY KEY,
                at REAL NOT NULL,
                op TEXT NOT NULL,
                record_id TEXT NOT NULL,
                content_sha TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                hash TEXT NOT NULL
            )
            """
        )
        self._migrate()
        self._conn.commit()
        self._data_version = self._current_data_version()
        if self._ann_threshold is not None:
            self._load()

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
        # An additive index keeps the database format and existing vectors intact. Reuse the
        # expression in search: SQLite only matches an expression index to the same expression.
        if self._sql_namespaces:
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS memories_vector_namespace "
                f"ON memories ({_NAMESPACE_SQL}) WHERE embedding IS NOT NULL"
            )
        # --- future migrations go here: `if cur < 2: <ALTER/UPDATE…>` then bump SCHEMA_VERSION ---
        if cur != SCHEMA_VERSION:
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def schema_version(self) -> int:
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def _load(self) -> None:
        for row in self._conn.execute(f"SELECT {_COLUMNS}, embedding FROM memories"):
            super().put(self._row_to_record(row))

    def _current_data_version(self) -> int:
        return int(self._conn.execute("PRAGMA data_version").fetchone()[0])

    def _refresh_if_stale(self) -> None:
        version = self._current_data_version()
        if version == self._data_version:
            return
        self._data_version = version
        if self._ann_threshold is None:
            self.version += 1
        else:
            InMemoryStore.clear(self)
            self._load()

    def refresh_version(self) -> int:
        """Refresh the change counter before reusing a derived lexical index."""
        with self._lock:
            self._refresh_if_stale()
            return self.version

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

    # ---- tamper-evident audit chain -------------------------------------------------------------

    def _audit(self, op: str, record_id: str, content_sha: str) -> None:
        """Append one hash-chained entry (caller holds the lock; committed with the mutation). Each
        hash covers the previous entry's hash, so editing or deleting ANY past row breaks every hash
        after it — `verify_audit_log` catches it. Multi-process safe: seq/prev are read and written
        inside the same transaction as the mutation."""
        row = self._conn.execute(
            "SELECT seq, hash FROM audit_log ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        prev_seq, prev_hash = (int(row[0]), str(row[1])) if row else (0, _AUDIT_GENESIS)
        seq, at = prev_seq + 1, time.time()
        self._conn.execute(
            "INSERT INTO audit_log (seq, at, op, record_id, content_sha, prev_hash, hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (seq, at, op, record_id, content_sha,
             prev_hash, _audit_hash(seq, at, op, record_id, content_sha, prev_hash)),
        )

    def audit_log(self, *, limit: int | None = None) -> list[dict]:
        """The mutation history (append-only): every put/delete/clear with its chain hashes — the
        compliance artifact behind "prove what happened to memory". No memory content, only hashes."""
        with self._lock:
            sql = "SELECT seq, at, op, record_id, content_sha, prev_hash, hash FROM audit_log " \
                  "ORDER BY seq"
            rows = self._conn.execute(sql).fetchall()
        entries = [dict(zip(("seq", "at", "op", "record_id", "content_sha", "prev_hash", "hash"), r))
                   for r in rows]
        return entries[-limit:] if limit else entries

    def verify_audit_log(self) -> dict:
        """Walk the whole chain recomputing every hash. Returns {ok, entries, first_invalid_seq}:
        any edited, removed, or reordered entry breaks the chain at the first affected seq."""
        entries = self.audit_log()
        prev_hash, expected_seq = _AUDIT_GENESIS, 1
        for e in entries:
            good = (
                e["seq"] == expected_seq
                and e["prev_hash"] == prev_hash
                and e["hash"] == _audit_hash(e["seq"], e["at"], e["op"], e["record_id"],
                                             e["content_sha"], e["prev_hash"])
            )
            if not good:
                return {"ok": False, "entries": len(entries), "first_invalid_seq": e["seq"]}
            prev_hash, expected_seq = e["hash"], expected_seq + 1
        return {"ok": True, "entries": len(entries), "first_invalid_seq": None}

    # ---- mutations (each writes its audit entry in the same transaction) -------------------------

    def put(self, record: MemoryRecord) -> None:
        self._put(record, replace=True)

    def put_if_absent(self, record: MemoryRecord) -> bool:
        """Insert a deterministic import ID once, even across concurrent imports."""
        return self._put(record, replace=False)

    def _put(self, record: MemoryRecord, *, replace: bool) -> bool:
        emb_blob = (struct.pack(f"<{len(record.embedding)}f", *record.embedding)
                    if record.embedding is not None else None)
        conflict = (
            """DO UPDATE SET content=excluded.content, kind=excluded.kind,
                importance=excluded.importance, source=excluded.source,
                provenance=excluded.provenance, actor=excluded.actor,
                metadata_json=excluded.metadata_json, created_at=excluded.created_at,
                updated_at=excluded.updated_at, superseded_by=excluded.superseded_by,
                embedding=excluded.embedding"""
            if replace else "DO NOTHING"
        )
        with self._lock:
            self._refresh_if_stale()
            with self._conn:
                cursor = self._conn.execute(
                    f"INSERT INTO memories ({_COLUMNS}, embedding) "
                    f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) {conflict}",
                    (record.id, record.content, record.kind, record.importance, record.source,
                     record.provenance, record.actor, json.dumps(record.metadata or {}),
                     record.created_at, record.updated_at, record.superseded_by, emb_blob),
                )
                changed = cursor.rowcount > 0
                if changed and self._audit_enabled:
                    self._audit("put", record.id, _content_sha(record))
            if changed:
                if self._ann_threshold is not None:
                    super().put(record)
                else:
                    self.version += 1
            return changed

    def get(self, record_id: str) -> MemoryRecord | None:
        with self._lock:
            self._refresh_if_stale()
            return self._get(record_id)

    def _get(self, record_id: str) -> MemoryRecord | None:
        row = self._conn.execute(
            f"SELECT {_COLUMNS}, embedding FROM memories WHERE id = ?", (record_id,),
        ).fetchone()
        return self._row_to_record(row) if row else None

    def iter_records(self, *, embeddings: bool = True, embedded_only: bool = False
                     ) -> Iterator[MemoryRecord]:
        """Stream records in insertion order. Metadata views can skip vector BLOBs entirely."""
        with self._lock:
            self._refresh_if_stale()
            column = "embedding" if embeddings else "NULL"
            where = " WHERE embedding IS NOT NULL" if embedded_only else ""
            cursor = self._conn.execute(f"SELECT {_COLUMNS}, {column} FROM memories{where} ORDER BY rowid")
            try:
                for row in cursor:
                    yield self._row_to_record(row)
            finally:
                cursor.close()

    def all(self) -> list[MemoryRecord]:
        return list(self.iter_records())

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT count(*) FROM memories").fetchone()[0])

    def embedding_dimensions(self) -> set[int]:
        with self._lock:
            return {int(row[0]) for row in self._conn.execute(
                "SELECT DISTINCT length(embedding) / 4 FROM memories WHERE embedding IS NOT NULL"
            )}

    def has_only_embedding_dimension(self, dimension: int) -> bool:
        """Identify a legacy vector width without decoding vectors or scanning mismatched stores."""
        size = dimension * 4
        with self._lock:
            return bool(self._conn.execute(
                """SELECT CASE
                   WHEN (SELECT length(embedding) FROM memories WHERE embedding IS NOT NULL LIMIT 1) = ?
                   THEN NOT EXISTS (
                       SELECT 1 FROM memories WHERE embedding IS NOT NULL AND length(embedding) != ?
                   )
                   ELSE 0 END""",
                (size, size),
            ).fetchone()[0])

    def search(
        self,
        embedding: list[float],
        *,
        limit: int,
        predicate: Callable[[MemoryRecord], bool] | None = None,
        allowed_ids: Collection[str] | None = None,
        namespace: str | None = None,
    ) -> list[tuple[float, MemoryRecord]]:
        if limit <= 0 or (allowed_ids is not None and not allowed_ids):
            return []
        with self._lock:
            self._refresh_if_stale()
            if namespace is not None and (not self._sql_namespaces or self._ann_threshold is not None):
                base = predicate
                predicate = lambda r: r.metadata.get("namespace") == namespace and (base is None or base(r))
            if self._ann_threshold is not None:
                return super().search(embedding, limit=limit, predicate=predicate, allowed_ids=allowed_ids)
            sql_namespace = namespace if self._sql_namespaces else None
            if predicate is not None:
                return self._search_exact(embedding, limit, predicate, allowed_ids, sql_namespace)
            # The vector scan and the top-k detail lookup must see the same committed data.
            own_snapshot = not self._conn.in_transaction
            with self._conn if own_snapshot else nullcontext():
                if own_snapshot:
                    self._conn.execute("BEGIN")
                hits = self._search_exact(embedding, limit, None, allowed_ids, sql_namespace)
                return [(score, record) for score, record_id in hits
                        if (record := self._get(record_id)) is not None]

    def search_namespace(self, embedding, *, namespace: str, limit: int, predicate=None):
        return self.search(embedding, limit=limit, predicate=predicate, namespace=namespace)

    def _search_exact(self, embedding, limit, predicate, allowed_ids, namespace):
        return heapq.nlargest(
            limit, self._scored_records(embedding, predicate, allowed_ids, namespace),
            key=lambda hit: hit[0],
        )

    def _scored_records(self, embedding, predicate, allowed_ids, namespace):
        allowed = set(allowed_ids) if allowed_ids is not None else None
        fields = _COLUMNS if predicate is not None else "id"
        sql = f"SELECT {fields}, embedding FROM memories WHERE embedding IS NOT NULL"
        params = []
        if namespace is not None:
            sql += f" AND {_NAMESPACE_SQL} = ?"
            params.append(namespace)
        if allowed is not None and len(allowed) <= _SQL_VARIABLE_LIMIT:
            sql += " AND id IN (" + ",".join("?" for _ in allowed) + ")"
            params.extend(allowed)
            allowed = None
        cursor = self._conn.execute(sql + " ORDER BY rowid", params)
        np = _numpy()
        query = np.asarray(embedding, dtype=np.float32) if np is not None else embedding
        try:
            while rows := cursor.fetchmany(_SEARCH_BATCH):
                candidates = []
                for row in rows:
                    if allowed is not None and row[0] not in allowed:
                        continue
                    if predicate is None:
                        candidates.append((row[0], row[-1]))
                    else:
                        record = self._row_to_record(row)
                        if predicate(record):
                            candidates.append((record, row[-1]))
                if not candidates:
                    continue
                dimension = len(query)
                if np is not None and dimension and all(len(blob) == 4 * dimension for _, blob in candidates):
                    matrix = np.frombuffer(b"".join(blob for _, blob in candidates), dtype="<f4")
                    matrix = matrix.reshape(-1, dimension)
                    scores = np.clip(np.einsum("ij,j->i", matrix, query), -1.0, 1.0)
                    yield from ((float(score), target) for score, (target, _) in zip(scores, candidates))
                else:
                    # Older databases can contain mixed dimensions from different embedders.
                    for target, blob in candidates:
                        yield cosine(embedding, struct.unpack(f"<{len(blob) // 4}f", blob)), target
        finally:
            cursor.close()


    def delete(self, record_id: str) -> bool:
        with self._lock:
            self._refresh_if_stale()
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    f"SELECT {_COLUMNS}, NULL FROM memories WHERE id = ?", (record_id,),
                ).fetchone()
                if row is None:
                    return False
                self._conn.execute("DELETE FROM memories WHERE id = ?", (record_id,))
                if self._audit_enabled:
                    self._audit("delete", record_id, _content_sha(self._row_to_record(row)))
            if self._ann_threshold is not None:
                super().delete(record_id)
            else:
                self.version += 1
            return True

    def clear(self) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute("DELETE FROM memories")
                if self._audit_enabled:
                    self._audit("clear", "*", _AUDIT_GENESIS)
            super().clear()

    def clear_namespace(self, namespace: str) -> int:
        """Clear one project's records and audit each deletion in one transaction."""
        with self._lock:
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                records = [r for r in self.iter_records(embeddings=False)
                           if r.metadata.get("namespace") == namespace]
                if not records:
                    return 0
                self._conn.executemany("DELETE FROM memories WHERE id = ?", ((r.id,) for r in records))
                if self._audit_enabled:
                    for record in records:
                        self._audit("delete", record.id, _content_sha(record))
            if self._ann_threshold is not None:
                for record in records:
                    super().delete(record.id)
            else:
                self.version += 1
            return len(records)

    def adopt_namespace(self, legacy: str, namespace: str, name: str, ids: set[str]) -> int:
        """Reassign explicitly selected legacy records without rewriting blobs or breaking chains."""
        with self._lock:
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                records = {r.id: r for r in self.iter_records(embeddings=False)}
                for rid in ids:
                    if rid not in records or records[rid].metadata.get("namespace") not in (legacy, namespace):
                        raise ValueError(f"{rid} is not in the selected legacy namespace.")
                for record in records.values():
                    for linked in (record.superseded_by, record.metadata.get("closes")):
                        if not isinstance(linked, str) or linked not in records:
                            continue
                        if (record.id in ids) == (linked in ids):
                            continue
                        outside = linked if record.id in ids else record.id
                        if records[outside].metadata.get("namespace") != namespace:
                            raise ValueError(f"Also select linked record {outside} before moving this chain.")
                moved = 0
                for rid in sorted(ids):
                    record = records[rid]
                    if record.metadata.get("namespace") == namespace:
                        continue
                    record.metadata.update(namespace=namespace, project_scope_name=name,
                                           legacy_namespace=legacy)
                    self._conn.execute("UPDATE memories SET metadata_json = ? WHERE id = ?",
                                       (json.dumps(record.metadata), rid))
                    if self._audit_enabled:
                        self._audit("scope", rid, _content_sha(record))
                    moved += 1
            if moved:
                if self._ann_threshold is not None:
                    super().clear()
                    self._load()
                else:
                    self.version += 1
            return moved

    def close(self) -> None:
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None
