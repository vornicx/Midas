import argparse
import json
import shutil
import subprocess

import pytest

from midas import HashingEmbedder, Memory, cli
from midas.project_scope import project_scope
from midas.sqlite_store import SQLiteStore


def test_same_named_repos_are_distinct_and_linked_worktrees_share_scope(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("Git is required for a worktree test")
    main = tmp_path / "one" / "app"
    other = tmp_path / "two" / "app"
    worktree = tmp_path / "feature"
    for path in (main, other):
        path.mkdir(parents=True)
        subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=Midas", "-c", "user.email=midas@example.test",
                    "commit", "--allow-empty", "-m", "initial"],
                   cwd=main, check=True, capture_output=True)
    subprocess.run(["git", "worktree", "add", "--detach", str(worktree)],
                   cwd=main, check=True, capture_output=True)
    subdir = worktree / "nested"
    subdir.mkdir()

    first = project_scope(main)
    second = project_scope(other)
    linked = project_scope(subdir)
    assert first.name == second.name == linked.name == "app"
    assert first.namespace == linked.namespace
    assert first.namespace != second.namespace
    assert linked.legacy_namespace == "feature"


def test_legacy_scope_adoption_preserves_vectors_history_and_unselected_records(tmp_path, capsys):
    path = tmp_path / "app"
    path.mkdir()
    db = tmp_path / "memory.sqlite3"
    store = SQLiteStore(db)
    mem = Memory(store=store, embedder=HashingEmbedder())
    old = mem.remember("Release date was March 9.", metadata={"namespace": "app"})
    new = mem.remember("Release date moved to April 10.", metadata={"namespace": "app"})
    unrelated = mem.remember("A record from the other app.", metadata={"namespace": "app"})
    old.superseded_by = new.id
    store.put(old)
    args = argparse.Namespace(directory=str(path), db=str(db), adopt_legacy="app", json=True,
                              apply=False, record=None)
    assert cli.cmd_scope(args) == 0
    assert len(json.loads(capsys.readouterr().out)["records"]) == 3
    assert store.get(old.id).metadata["namespace"] == "app"

    args.apply = True
    args.record = [old.id]
    assert cli.cmd_scope(args) == 1
    assert new.id in capsys.readouterr().err
    assert store.get(old.id).metadata["namespace"] == "app"

    args.record = [old.id, new.id]
    assert cli.cmd_scope(args) == 0
    assert json.loads(capsys.readouterr().out)["moved"] == 2
    moved = store.get(old.id)
    assert moved.metadata["namespace"] == project_scope(path).namespace
    assert list(moved.embedding) == pytest.approx(list(old.embedding))
    assert moved.superseded_by == new.id
    assert store.get(unrelated.id).metadata["namespace"] == "app"
    assert store.verify_audit_log()["ok"]
    store.close()


def test_mcp_auto_scope_filters_lookup_capture_maintenance_and_delete(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from midas import mcp_server

    left, right = (tmp_path / team / "app" for team in ("one", "two"))
    left.mkdir(parents=True)
    right.mkdir(parents=True)
    store = SQLiteStore(tmp_path / "scoped.sqlite3")
    mem = Memory(store=store, embedder=HashingEmbedder(), supersede=True)
    monkeypatch.setattr(mcp_server, "_memory", lambda: mem)
    context = {"client": "codex", "actor": "", "namespace": "auto"}

    def switch(directory):
        return mcp_server._session.set(mcp_server.session_context({**context, "cwd": str(directory)}))

    content = "The PostgreSQL production database was migrated to version 16."
    token = switch(left)
    try:
        first = mcp_server.capture(content, kind="fact")
        assert first["stored"]
    finally:
        mcp_server._session.reset(token)
    token = switch(right)
    try:
        second = mcp_server.capture(content, kind="fact")
        assert second["stored"] and first["id"] != second["id"]
        assert mcp_server.inspect_memory(first["id"])["found"] is False
        assert "no memory" in mcp_server.forget(first["id"])
        assert len(mcp_server.stats()["by_namespace"]) == 1
        with pytest.raises(ValueError, match="scoped"):
            mcp_server.recall("PostgreSQL", namespace=project_scope(left).namespace)
        assert mcp_server.maintain(max_records=100)["before"] == 1
        assert mcp_server.forget_all() == "cleared 1 memories"
    finally:
        mcp_server._session.reset(token)
    assert store.get(first["id"]) is not None and store.get(second["id"]) is None
    store.close()
