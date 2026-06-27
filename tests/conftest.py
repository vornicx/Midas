"""Suite-wide guards.

The MCP server builds its store at import time, and the default is now a persistent ~/.midas store so a
fresh install just works. Tests must NEVER touch that real store — pin an ephemeral in-memory store for
the whole suite (set before any test imports midas.mcp_server)."""
import os

os.environ.setdefault("MIDAS_MCP_DB", ":memory:")
# A bare Memory() now auto-selects LocalEmbedder when fastembed is present; pin hashing so the suite
# stays deterministic and fast (no model load) regardless of the dev env.
os.environ.setdefault("MIDAS_EMBEDDER", "hashing")
