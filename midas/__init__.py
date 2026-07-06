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
from .guard import (
    MEMORY_USES,
    Armorer,
    EvidenceRef,
    Guard,
    MemoryUse,
    MemoryUseDecision,
    ProvenanceStamp,
    decide_memory_use,
)
from .continuity import (
    Conflict,
    close_loop,
    memory_conflicts,
    open_loops,
    remember_commitment,
    resume,
)
from .distill import Distiller, HTTPDistiller, OllamaDistiller
from .importance import ContentImportance, StructuralImportance, is_standing_instruction
from .memory import CaptureResult, ContextBlock, Memory, Reranker, approx_tokens, format_record
from .policy import (
    AGENT_MEMORY_INSTRUCTIONS,
    DEFAULT_POLICY,
    MemoryPolicy,
    parse_ttl_spec,
    policy_summary,
)
from .index import MemoryStore, VectorIndex
from .store import InMemoryStore
from .types import MEMORY_KINDS, MEMORY_PROVENANCE, MemoryKind, MemoryProvenance, MemoryRecord, RecallHit

try:
    from .sqlite_store import SQLiteStore
except ImportError:
    SQLiteStore = None  # sqlite-vec not installed

try:
    from .ann import IVFIndex, IVFStore
except ImportError:
    IVFIndex = IVFStore = None  # numpy not installed (ANN backend is optional)

try:
    from .turbovec_index import TurboVecIndex
    from .turbovec_store import TurboVecStore
except ImportError:
    TurboVecIndex = TurboVecStore = None  # `pip install turbovec` not present (compressed backend is optional)

__all__ = [
    "Memory",
    "Armorer",
    "Guard",
    "ProvenanceStamp",
    "EvidenceRef",
    "MemoryUseDecision",
    "MemoryUse",
    "MEMORY_USES",
    "decide_memory_use",
    "ContentImportance",
    "StructuralImportance",
    "is_standing_instruction",
    "Distiller",
    "OllamaDistiller",
    "HTTPDistiller",
    "MemoryPolicy",
    "DEFAULT_POLICY",
    "AGENT_MEMORY_INSTRUCTIONS",
    "policy_summary",
    "parse_ttl_spec",
    "Conflict",
    "memory_conflicts",
    "resume",
    "open_loops",
    "remember_commitment",
    "close_loop",
    "CaptureResult",
    "ContextBlock",
    "Reranker",
    "approx_tokens",
    "format_record",
    "MemoryRecord",
    "RecallHit",
    "MemoryKind",
    "MEMORY_KINDS",
    "MemoryProvenance",
    "MEMORY_PROVENANCE",
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
    "MemoryStore",
    "VectorIndex",
    "SQLiteStore",
    "IVFIndex",
    "IVFStore",
]
__version__ = "1.1.0"
