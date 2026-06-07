"""midas-memory-mcp — launcher for the Midas MCP server.

This package contains no memory logic of its own. It declares a dependency on the Midas SDK
(`midas-memory`) with the MCP + local-embedding extras and exposes the `midas-memory-mcp` console
command, so that `uvx midas-memory-mcp` and `pip install midas-memory-mcp` Just Work for any MCP
client (Claude Code, Cursor, Codex, Windsurf, …).

All functionality lives in `midas`. See https://github.com/vornicx/Midas
"""
from __future__ import annotations

__version__ = "0.0.1"
__all__ = ["main"]


def main() -> None:
    """Run the Midas MCP server (delegates to the SDK's entry point)."""
    from midas.mcp_server import main as _run

    _run()
