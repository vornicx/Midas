import argparse
import json

from midas import HashingEmbedder, Memory, cli
from midas.native_import import load_native
from midas.project_scope import project_scope
from midas.sqlite_store import SQLiteStore


def test_claude_topics_and_codex_v1_v2_use_relevant_markdown(tmp_path):
    claude = tmp_path / "memory"
    claude.mkdir()
    (claude / "MEMORY.md").write_text("- [Deployment](deploy.md) — topic index.\n", encoding="utf-8")
    (claude / "deploy.md").write_text(
        "---\nname: Deployment\ntype: project\nmodified: 2026-11-24\n---\n"
        "Production runs PostgreSQL 16.\n", encoding="utf-8",
    )
    [topic] = load_native(claude, "claude-memory")
    assert topic.kind == "fact" and topic.line == 6 and topic.file.endswith("deploy.md")
    assert topic.group == "Deployment"

    for name in ("memories", "memories_v2"):
        codex = tmp_path / name
        codex.mkdir()
        (codex / "raw_memories.md").write_text("Never import this raw memory.", encoding="utf-8")
        (codex / "MEMORY.md").write_text(
            "# Task Group: Deployments\nscope: service\napplies_to: cwd=sample-app\n\n"
            "## Task 1: Fix worker\n### keywords\n- random-keyword to skip\n"
            "## User preferences\n- Prefer staging during release validation.\n"
            "## Reusable knowledge\n- Production runs PostgreSQL 16.\n", encoding="utf-8",
        )
        pref, fact = load_native(codex, "codex-memory")
        assert pref.kind == "preference" and pref.line == 9
        assert pref.origin_scope == "cwd=sample-app" and pref.group == "Deployments"
        assert fact.kind == "fact"


def test_selected_native_import_is_read_only_until_apply_and_retries_safely(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MIDAS_MCP_EMBEDDER", "hashing")
    db = tmp_path / "memory.sqlite3"
    source = tmp_path / "MEMORY.md"
    source.write_text("# Reusable knowledge\n- Queue consumers use Redis Streams.\n"
                      "- Deployments go through staging first.\n", encoding="utf-8")
    args = argparse.Namespace(file=str(source), source="codex-memory", db=str(db), json=True,
                              into=str(tmp_path), apply=False, select=None)
    assert cli.cmd_import(args) == 0
    preview = json.loads(capsys.readouterr().out)
    assert not db.exists() and len(preview["items"]) == 2

    args.apply = True
    assert cli.cmd_import(args) == 1 and not db.exists()
    assert "Select" in capsys.readouterr().err
    args.select = [preview["items"][0]["selection"]]
    assert cli.cmd_import(args) == 0
    assert json.loads(capsys.readouterr().out)["imported"] == 1
    assert cli.cmd_import(args) == 0
    assert json.loads(capsys.readouterr().out)["skipped"] == 1

    mem = Memory(store=SQLiteStore(db), embedder=HashingEmbedder())
    [record] = mem.store.all()
    assert record.provenance == "observation" and record.actor == "codex"
    assert record.source.endswith("MEMORY.md:2") and record.metadata["imported_line"] == 2
    assert mem.recall("Redis Streams", metadata_filter={"namespace": project_scope(tmp_path).namespace})
    assert mem.store.verify_audit_log()["entries"] == 1
    mem.store.close()

    source.write_text("# Reusable knowledge\n- Queue consumers use RabbitMQ now.\n", encoding="utf-8")
    assert cli.cmd_import(args) == 1
    assert "Unknown or changed selection" in capsys.readouterr().err
