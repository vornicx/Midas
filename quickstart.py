"""Midas quickstart — the public SDK surface.

Runs offline out of the box (HashingEmbedder). For real semantic quality install the
local extra (`uv pip install fastembed`) and swap in LocalEmbedder + LocalReranker:

    from midas import LocalEmbedder, LocalReranker
    mem = Memory(embedder=LocalEmbedder(), reranker=LocalReranker())

Run: `uv run --no-sync python quickstart.py`
"""
from midas import Memory

# Zero-setup: in-memory store + offline hashing embedder.
mem = Memory()

# Remember turns/facts. `kind` + `importance` (1-5) feed the relevance×importance×recency blend;
# `metadata` carries a thread key so retrieval can pull in neighbouring turns.
thread = {"session": "project-x"}
mem.remember("Decision: the primary database is PostgreSQL.", kind="constraint", importance=5, metadata=thread)
mem.remember("We chatted about the weather for a while.", kind="chat", metadata=thread)
mem.remember("The launch date moved to September 14.", kind="fact", importance=5, metadata=thread)
mem.remember("Reminder to buy more coffee for the office.", kind="chat", metadata=thread)

# Assemble a budgeted, prompt-ready context block. `window` pulls in same-thread neighbours;
# the block is ordered highest-value-first so it sits at the top of a prompt, not buried.
print("--- assembled context ---")
print(mem.assemble("When do we launch?", token_budget=128, window=1, thread_key="session"))

# Or get structured, ranked hits to use however you like.
print("\n--- ranked recall ---")
for hit in mem.recall("which database did we pick?", limit=3):
    print(f"  {hit.score:.2f}  {hit.record.content}")
