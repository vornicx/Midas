"""Exercise the same stdio entry point and detached worker used by installed clients."""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path

import pytest
from midas.project_scope import project_scope


pytest.importorskip("mcp")

ROOT = Path(__file__).resolve().parents[1]


async def request(process, method, params, *, request_id=1):
    message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
    process.stdin.write(json.dumps(message).encode() + b"\n")
    await process.stdin.drain()
    reply = json.loads(await asyncio.wait_for(process.stdout.readline(), 10))
    assert reply["id"] == request_id and "error" not in reply, reply
    return reply["result"]


async def initialize(process):
    await request(process, "initialize", {
        "protocolVersion": "2024-11-05", "capabilities": {},
        "clientInfo": {"name": "midas-integration-test", "version": "1"},
    })
    process.stdin.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
    await process.stdin.drain()


async def tool(process, name, **arguments):
    result = await request(process, "tools/call", {"name": name, "arguments": arguments})
    assert not result.get("isError"), result
    return [part["text"] for part in result["content"]]


def test_concurrent_stdio_clients_share_worker_and_live_prompt_capture(tmp_path):
    db = tmp_path / "database.sqlite3"
    project = tmp_path / "project"
    project.mkdir()
    env = {
        **os.environ, "HOME": str(tmp_path), "USERPROFILE": str(tmp_path), "PYTHONPATH": str(ROOT),
        "MIDAS_MCP_DB": str(db), "MIDAS_MCP_EMBEDDER": "hashing", "MIDAS_MCP_MODE": "shared",
        "MIDAS_MCP_NAMESPACE": "", "MIDAS_MCP_ACTOR": "",
    }
    processes = []
    worker_pids = set()

    async def start(client, namespace="", extra=()):
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "from midas_mcp import main; main()", *extra,
            cwd=project, env={**env, "MIDAS_MCP_CLIENT": client, "MIDAS_MCP_NAMESPACE": namespace},
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        )
        processes.append(proc)
        return proc

    def worker_pid():
        [endpoint] = (tmp_path / ".midas" / "run").glob("*.json")
        pid = json.loads(endpoint.read_text())["pid"]
        worker_pids.add(pid)
        return pid

    async def check():
        try:
            scope, reader, other = await asyncio.gather(
                start("codex", "auto"), start("claude-code"), start("cursor", "another-project"),
            )
            await asyncio.gather(*(initialize(p) for p in (scope, reader, other)))
            pid = worker_pid()

            await asyncio.gather(
                tool(scope, "remember", content="PostgreSQL is the production database.",
                     kind="fact", importance=5),
                tool(reader, "remember", content="Redis stores the queue.", kind="fact", importance=5),
            )
            [record] = await tool(reader, "recall", query="PostgreSQL", limit=1)
            hit = json.loads(record)
            assert hit["source"].startswith("mcp:codex:") and hit["actor"] == "codex"
            assert hit["metadata"]["namespace"] == project_scope(project).namespace
            assert hit["metadata"]["project_scope_name"] == "project"
            assert hit["metadata"]["origin"]["cwd"] == "project"
            assert await tool(other, "recall", query="PostgreSQL") == []

            hook = await start("agent", extra=("--capture-prompt", "codex"))
            await hook.communicate(json.dumps({
                "prompt": "The Atlas production API rate limit is exactly 5000 requests per minute.",
                "session_id": "codex-thread", "cwd": str(project),
            }).encode())
            assert hook.returncode == 0
            for _ in range(40):
                hits = await tool(reader, "recall", query="Atlas production API rate limit", limit=1)
                if hits and json.loads(hits[0])["source"] == "hook:codex:codex-thread":
                    break
                await asyncio.sleep(0.05)
            else:
                pytest.fail("the live prompt did not reach the shared store")

            scope.stdin.close()
            await asyncio.wait_for(scope.wait(), 5)
            assert worker_pid() == pid
            assert await tool(reader, "recall", query="PostgreSQL", limit=1)

            # The worker can restart using the existing DB even if killed before cleaning its endpoint.
            os.kill(pid, signal.SIGTERM)
            worker_pids.remove(pid)
            await asyncio.wait_for(reader.wait(), 5)
            reopened = await start("codex")
            await initialize(reopened)
            assert worker_pid() != pid
            assert await tool(reopened, "recall", query="PostgreSQL", limit=1)
        finally:
            for process in processes:
                if process.returncode is None:
                    process.stdin.close()
                    await asyncio.wait_for(process.wait(), 5)
            for pid in worker_pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    asyncio.run(asyncio.wait_for(check(), 30))
