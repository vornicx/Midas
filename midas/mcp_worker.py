"""One local MCP engine, with independent protocol sessions over loopback sockets."""

import asyncio
import hmac
import json
import os
import secrets
from pathlib import Path

from mcp.server.stdio import stdio_server

from midas import mcp_server
from midas_mcp import MAX_MESSAGE_BYTES, PROTOCOL, engine_fingerprint


IDLE_SECONDS = 30


class _Input:
    def __init__(self, reader):
        self.reader = reader

    async def __aiter__(self):
        while line := await self.reader.readline():
            yield line.decode("utf-8", errors="replace")


class _Output:
    def __init__(self, writer):
        self.writer = writer

    async def write(self, line):
        self.writer.write(line.encode("utf-8"))

    async def flush(self):
        await self.writer.drain()


async def _serve(endpoint: Path) -> None:
    token = secrets.token_urlsafe(32)
    engine = engine_fingerprint()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    idle_timer = loop.call_later(IDLE_SECONDS, stop.set)
    connections = 0

    async def serve_client(reader, writer):
        nonlocal connections, idle_timer
        connections += 1
        idle_timer.cancel()
        try:
            raw = await asyncio.wait_for(reader.readline(), 3)
            if len(raw) > 16384:
                return
            hello = json.loads(raw)
            if not isinstance(hello, dict) or not hmac.compare_digest(
                    str(hello.get("token", "")).encode(), token.encode()):
                return
            capture = hello.get("capture") is True
            if hello.get("protocol") != PROTOCOL or (not capture and hello.get("engine") != engine):
                writer.write(b"!")
                await writer.drain()
                return
            writer.write(b"+")
            await writer.drain()
            if capture:
                event = json.loads(await asyncio.wait_for(reader.readline(), 3))
                await mcp_server.capture_prompt(event)
                return
            session = await asyncio.to_thread(mcp_server.session_context, hello)
            context = mcp_server._session.set(session)
            try:
                async with stdio_server(_Input(reader), _Output(writer)) as (incoming, outgoing):
                    server = mcp_server.server._mcp_server
                    await server.run(incoming, outgoing, server.create_initialization_options())
            finally:
                mcp_server._session.reset(context)
        except (OSError, ValueError, TimeoutError, KeyError, TypeError):
            pass  # A failed handshake or disconnected client cannot interrupt other sessions.
        finally:
            writer.close()
            connections -= 1
            if not connections:
                idle_timer = loop.call_later(IDLE_SECONDS, stop.set)

    server = await asyncio.start_server(serve_client, "127.0.0.1", 0, limit=MAX_MESSAGE_BYTES)
    async with server:
        temporary = endpoint.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps({
                "port": server.sockets[0].getsockname()[1], "token": token, "pid": os.getpid(),
            }), encoding="utf-8")
            temporary.replace(endpoint)
            await stop.wait()
        finally:
            endpoint.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
            idle_timer.cancel()


def run_worker(endpoint: Path) -> None:
    mcp_server._start_maintenance()
    asyncio.run(_serve(endpoint))
