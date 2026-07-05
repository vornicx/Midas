"""Encryption at rest (opt-in SQLCipher): the file is unreadable without the key, everything else —
search, audit chain, persistence — behaves identically. Skipped when the `encrypted` extra is absent."""
from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("sqlcipher3")

from midas import HashingEmbedder, Memory
from midas.sqlite_store import SQLiteStore


def test_encrypted_store_roundtrip_and_ciphertext(tmp_path) -> None:
    db = str(tmp_path / "m.sqlite3")
    mem = Memory(store=SQLiteStore(db, key="s3cret"), embedder=HashingEmbedder())
    mem.remember("The primary database is PostgreSQL.", kind="constraint", importance=5)

    # plaintext never touches disk
    raw = (tmp_path / "m.sqlite3").read_bytes()
    assert b"PostgreSQL" not in raw and not raw.startswith(b"SQLite format 3")

    # right key: reopen and recall; audit chain intact
    again = SQLiteStore(db, key="s3cret")
    assert any("PostgreSQL" in r.content for r in again.all())
    assert again.verify_audit_log()["ok"]

    # plain sqlite (no key) cannot read the file
    with pytest.raises(sqlite3.DatabaseError):
        sqlite3.connect(db).execute("SELECT count(*) FROM memories").fetchone()

    # wrong key fails too (surfaced as a database error on open)
    with pytest.raises(Exception):
        SQLiteStore(db, key="wrong")


def test_key_from_env(tmp_path, monkeypatch) -> None:
    db = str(tmp_path / "env.sqlite3")
    monkeypatch.setenv("MIDAS_MCP_KEY", "env-key")
    SQLiteStore(db)  # picks the key up from the environment
    monkeypatch.delenv("MIDAS_MCP_KEY")
    with pytest.raises(Exception):
        SQLiteStore(db).all()  # without the key it is not a readable database
