"""MCPB entry point for the Midas memory connector.

The packaged manifest runs Midas via `uvx midas-memory-mcp`, which resolves and installs the
published `midas-memory[mcp,local]` package from PyPI (cross-platform, no vendored native wheels).
This file is the bundle's declared entry point and a fallback: if the Midas MCP server package is
already importable, `python server/main.py` runs it directly.

Source: https://github.com/vornicx/Midas
"""
from __future__ import annotations


def main() -> None:
    from midas.mcp_server import main as run

    run()


if __name__ == "__main__":
    main()
