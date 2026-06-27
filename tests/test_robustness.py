"""Core robustness under stress: concurrent threaded writes, cross-instance visibility on one SQLite
file (the multi-client promise), weird/large inputs, and recall over a large store. Deterministic —
HashingEmbedder, no LLM."""
from __future__ import annotations

import threading

import pytest

from midas import HashingEmbedder, Memory
from midas.sqlite_store import SQLiteStore


def test_concurrent_threaded_writes_all_land(tmp_path) -> None:
    store = SQLiteStore(str(tmp_path / "concurrent.sqlite3"))
    mem = Memory(store=store, embedder=HashingEmbedder())

    def worker(n: int) -> None:
        for i in range(50):
            mem.remember(f"thread {n} fact {i}", kind="fact")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store.all()) == 200  # 4 threads x 50, no lost or corrupted writes


def test_cross_instance_visibility_on_one_file(tmp_path) -> None:
    path = str(tmp_path / "shared.sqlite3")
    a = Memory(store=SQLiteStore(path), embedder=HashingEmbedder())
    b = Memory(store=SQLiteStore(path), embedder=HashingEmbedder())
    a.remember("Apollo uses PostgreSQL.", kind="constraint", importance=5)
    # b sees a's write without a restart (SQLite data_version refresh) — the multi-client guarantee
    assert any("PostgreSQL" in r.content for r in b.store.all())


def test_weird_inputs_do_not_crash(tmp_path) -> None:
    mem = Memory(store=SQLiteStore(str(tmp_path / "weird.sqlite3")), embedder=HashingEmbedder())
    inputs = ["café ☕ 日本語 한국어 🚀", "x" * 20_000, "😀" * 100, "tabs\tand\nnewlines"]
    for content in inputs:
        mem.remember(content, kind="note")
    assert len(mem.store.all()) == len(inputs)
    assert isinstance(mem.recall("日本語"), list)        # unicode query — no crash
    assert isinstance(mem.recall("x" * 5_000), list)     # huge query — no crash
    # empty / whitespace-only content is intentionally rejected (a robustness feature, not a crash)
    with pytest.raises(ValueError):
        mem.remember("   ", kind="note")


def test_recall_finds_the_signal_in_a_large_store(tmp_path) -> None:
    mem = Memory(store=SQLiteStore(str(tmp_path / "big.sqlite3")), embedder=HashingEmbedder())
    for i in range(1500):
        mem.remember(f"log line {i} about routine topic {i % 40}", kind="note", importance=2)
    mem.remember("Decision: the primary database is PostgreSQL.", kind="constraint", importance=5)
    hits = mem.recall("which database did we pick for the primary store", limit=3)
    assert hits and "PostgreSQL" in hits[0].record.content  # the signal ranks top among 1500 noise records
