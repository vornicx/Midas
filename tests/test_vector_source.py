"""SQLiteVectorSource — on-disk id→embedding store for fetch-by-id rerank (Phase C, RAM-saving)."""
from __future__ import annotations

from midas.vector_source import SQLiteVectorSource


def test_roundtrip_and_batch_get(tmp_path):
    vs = SQLiteVectorSource(tmp_path / "v.db")
    vs.add_many(["a", "b", "c"], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    got = vs.get_many(["a", "c", "missing"])
    assert got["a"] == [1.0, 2.0]
    assert got["c"] == [5.0, 6.0]
    assert "missing" not in got          # absent ids are simply omitted
    assert len(vs) == 3


def test_upsert_overwrites(tmp_path):
    vs = SQLiteVectorSource(tmp_path / "v.db")
    vs.add_many(["a"], [[1.0, 1.0]])
    vs.add_many(["a"], [[9.0, 9.0]])     # same id → overwrite, not duplicate
    assert vs.get_many(["a"])["a"] == [9.0, 9.0]
    assert len(vs) == 1


def test_remove(tmp_path):
    vs = SQLiteVectorSource(tmp_path / "v.db")
    vs.add_many(["a", "b"], [[1.0], [2.0]])
    assert vs.remove("a") is True
    assert vs.remove("a") is False       # idempotent
    assert vs.get_many(["a"]) == {}
    assert len(vs) == 1


def test_persistence_across_reopen(tmp_path):
    p = tmp_path / "v.db"
    vs = SQLiteVectorSource(p)
    vs.add_many(["x"], [[0.5, 0.5, 0.5]])
    vs.close()
    reopened = SQLiteVectorSource(p)
    assert reopened.get_many(["x"])["x"] == [0.5, 0.5, 0.5]


def test_large_batch_exceeds_sqlite_var_limit(tmp_path):
    vs = SQLiteVectorSource(tmp_path / "v.db")
    ids = [f"r{i}" for i in range(2500)]   # > the 999-variable limit → must chunk
    vs.add_many(ids, [[float(i)] for i in range(2500)])
    got = vs.get_many(ids)
    assert len(got) == 2500 and got["r2499"] == [2499.0]
