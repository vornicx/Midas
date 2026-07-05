"""The tamper-evident audit chain: every SQLite mutation appends a hash-chained entry (no memory
content, only hashes), and `verify_audit_log` / `midas audit` detect any after-the-fact edit."""
from __future__ import annotations

import argparse
import json
import sqlite3

from midas import HashingEmbedder, Memory
from midas.sqlite_store import SQLiteStore


def _mem(path: str) -> Memory:
    return Memory(store=SQLiteStore(path), embedder=HashingEmbedder())


def test_mutations_append_chained_entries(tmp_path) -> None:
    mem = _mem(str(tmp_path / "m.sqlite3"))
    a = mem.remember("The primary database is PostgreSQL.", kind="constraint")
    mem.remember("The launch date is September 14.", kind="fact")
    mem.forget(a.id)

    log = mem.store.audit_log()
    assert [e["op"] for e in log] == ["put", "put", "delete"]
    assert log[0]["record_id"] == a.id and log[2]["record_id"] == a.id
    # the chain links: each entry's prev_hash is the previous entry's hash
    assert log[1]["prev_hash"] == log[0]["hash"] and log[2]["prev_hash"] == log[1]["hash"]
    # no memory content anywhere in the log (the audit boundary)
    assert "PostgreSQL" not in json.dumps(log)

    assert mem.store.verify_audit_log() == {"ok": True, "entries": 3, "first_invalid_seq": None}


def test_supersession_and_clear_are_audited(tmp_path) -> None:
    db = str(tmp_path / "m.sqlite3")
    mem = Memory(store=SQLiteStore(db), embedder=HashingEmbedder(), supersede=True)
    mem.remember("The launch date is September 14.", kind="fact")
    mem.remember("The launch date is September 20.", kind="fact")
    ops = [e["op"] for e in mem.store.audit_log()]
    assert ops.count("put") >= 3  # 2 inserts + the revised head re-written with its supersession link
    mem.store.clear()
    assert mem.store.audit_log()[-1]["op"] == "clear"
    assert mem.store.verify_audit_log()["ok"]


def test_tampering_breaks_the_chain(tmp_path) -> None:
    db = str(tmp_path / "m.sqlite3")
    mem = _mem(db)
    for i in range(4):
        mem.remember(f"note number {i} about service routing", kind="note")
    mem.store.close()

    con = sqlite3.connect(db)  # an attacker rewrites history directly
    con.execute("UPDATE audit_log SET content_sha = 'f' || substr(content_sha, 2) WHERE seq = 2")
    con.commit()
    con.close()

    result = SQLiteStore(db).verify_audit_log()
    assert result["ok"] is False and result["first_invalid_seq"] == 2


def test_deleting_an_entry_breaks_the_chain(tmp_path) -> None:
    db = str(tmp_path / "m.sqlite3")
    mem = _mem(db)
    for i in range(3):
        mem.remember(f"note number {i} about service routing", kind="note")
    mem.store.close()

    con = sqlite3.connect(db)  # covering tracks by removing the middle entry
    con.execute("DELETE FROM audit_log WHERE seq = 2")
    con.commit()
    con.close()

    result = SQLiteStore(db).verify_audit_log()
    assert result["ok"] is False and result["first_invalid_seq"] == 3  # the gap is where it breaks


def test_chain_survives_reopen_and_cross_instance(tmp_path) -> None:
    db = str(tmp_path / "m.sqlite3")
    _mem(db).remember("first fact about the deploy pipeline", kind="fact")
    _mem(db).remember("second fact about the deploy pipeline", kind="fact")  # a separate connection
    store = SQLiteStore(db)
    assert store.verify_audit_log() == {"ok": True, "entries": 2, "first_invalid_seq": None}
    assert [e["seq"] for e in store.audit_log()] == [1, 2]


def test_audit_can_be_disabled(tmp_path) -> None:
    store = SQLiteStore(str(tmp_path / "m.sqlite3"), audit=False)
    Memory(store=store, embedder=HashingEmbedder()).remember("perf-path write", kind="note")
    assert store.audit_log() == []


def test_cli_audit_command(tmp_path, capsys) -> None:
    from midas import cli

    db = str(tmp_path / "m.sqlite3")
    assert cli.cmd_audit(argparse.Namespace(db=db, limit=10, json=False)) == 1  # no store yet
    capsys.readouterr()

    _mem(db).remember("The primary database is PostgreSQL.", kind="constraint")
    assert cli.cmd_audit(argparse.Namespace(db=db, limit=10, json=True)) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["receipt_kind"] == "audit_chain" and out["ok"] is True and out["entries"] == 1
    assert "PostgreSQL" not in json.dumps(out)  # receipts never leak memory content

    con = sqlite3.connect(db)
    con.execute("UPDATE audit_log SET op = 'delete' WHERE seq = 1")
    con.commit()
    con.close()
    assert cli.cmd_audit(argparse.Namespace(db=db, limit=10, json=False)) == 1
    assert "BROKEN" in capsys.readouterr().out
