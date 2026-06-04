"""Demo: a coding agent that remembers across sessions — locally, with source provenance, no LLM.

The pain: an agent's context window resets between sessions, so it re-derives decisions and even
repeats dead ends. Midas gives it durable, source-traceable memory — with no LLM call at ingest or
query and nothing leaving the machine.

Run:  uv run --no-sync python examples/coding_agent_demo.py
      Install the `local` extra (`uv pip install -e ".[local]"`) for semantic recall; otherwise this
      falls back to an offline hashing embedder (lexical, zero-dependency).
"""
from __future__ import annotations

import os
import sys

from midas import HashingEmbedder, Memory


def build_memory() -> Memory:
    if os.getenv("MIDAS_DEMO_EMBEDDER", "local") != "hashing":
        try:
            from midas import LocalEmbedder, LocalReranker

            return Memory(embedder=LocalEmbedder(), reranker=LocalReranker())
        except Exception as exc:  # fastembed not installed
            print(f"(local embedder unavailable: {exc}; using offline hashing)\n", file=sys.stderr)
    return Memory(embedder=HashingEmbedder())


mem = build_memory()
thread = {"session": "build-acme-api"}

print("=== Session 1: the agent makes decisions, hits a dead end, gets corrected ===")
mem.remember("Decision: the primary datastore is PostgreSQL (need ACID + relational queries).",
             kind="constraint", importance=5, metadata=thread)
mem.remember("Tried Redis as the primary store — too volatile for our data, so we REVERTED it.",
             kind="note", importance=4, metadata=thread)
mem.remember("Auth is JWT with 15-minute access tokens and refresh-token rotation.",
             kind="constraint", importance=4, metadata=thread)
mem.remember("User correction: never commit secrets — load them from the environment.",
             kind="preference", importance=5, metadata=thread)
mem.remember("We chatted about lunch options for a bit.", kind="chat", importance=1, metadata=thread)
print("  5 turns remembered — no LLM call, nothing left the machine.\n")

print("=== Session 2 (fresh context window): the agent rehydrates project memory ===")
for q in [
    "what datastore are we using?",
    "should I use Redis as the main database?",
    "how do we handle secrets?",
]:
    print(f"\nQ: {q}")
    for hit in mem.recall(q, limit=2):
        print(f"   [{hit.record.kind} | src:{hit.record.id[:8]} | {hit.score:.2f}] {hit.record.content}")

print("\n=== Budgeted context block (drop straight into the agent's prompt) ===")
print(mem.assemble("summarize the current project decisions and constraints",
                   token_budget=160, window=1, thread_key="session"))
print("\nEvery hit traces to a source turn (src:…). 0 LLM calls at ingest or query.")
