"""Midas — agentic memory.

Open-core wedge: semantic recall + budgeted context assembly behind a small,
store/embedder-agnostic API. This is the thing the eval harness benchmarks.
"""
from .embeddings import (
    DiskCachedEmbedder,
    Embedder,
    HashingEmbedder,
    LocalEmbedder,
    LocalReranker,
    OpenAIEmbedder,
    configure_local_model_cache,
    cosine,
    tokenize,
)
from .importance import ContentImportance
from .memory import CaptureResult, ContextBlock, Memory, Reranker, approx_tokens, format_record
from .policy import AGENT_MEMORY_INSTRUCTIONS, DEFAULT_POLICY, MemoryPolicy, policy_summary
from .store import InMemoryStore
from .types import MEMORY_KINDS, MemoryKind, MemoryRecord, RecallHit

try:
    from .sqlite_store import SQLiteStore
except ImportError:
    SQLiteStore = None  # sqlite-vec not installed

try:
    from .ann import IVFIndex, IVFStore
except ImportError:
    IVFIndex = IVFStore = None  # numpy not installed (ANN backend is optional)

__all__ = [
    "Memory",
    "ContentImportance",
    "MemoryPolicy",
    "DEFAULT_POLICY",
    "AGENT_MEMORY_INSTRUCTIONS",
    "policy_summary",
    "CaptureResult",
    "ContextBlock",
    "Reranker",
    "approx_tokens",
    "format_record",
    "MemoryRecord",
    "RecallHit",
    "MemoryKind",
    "MEMORY_KINDS",
    "Embedder",
    "DiskCachedEmbedder",
    "HashingEmbedder",
    "LocalEmbedder",
    "LocalReranker",
    "OpenAIEmbedder",
    "configure_local_model_cache",
    "cosine",
    "tokenize",
    "InMemoryStore",
    "SQLiteStore",
    "IVFIndex",
    "IVFStore",
]
__version__ = "0.0.1"
