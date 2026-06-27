"""Midas quickstart — the public SDK surface, end to end.

Runs offline out of the box (HashingEmbedder). For real semantic quality install the local extra
(`uv pip install fastembed`) and swap in LocalEmbedder + LocalReranker:

    from midas import LocalEmbedder, LocalReranker
    mem = Memory(embedder=LocalEmbedder(), reranker=LocalReranker())

Run: `uv run --no-sync python quickstart.py`
"""
from midas import Memory
from midas.guard import decide_memory_use
from midas.state import memory_state

# Zero-setup: in-memory store + offline hashing embedder. supersede=True so revised beliefs replace stale.
mem = Memory(supersede=True)

# 1) Remember turns/facts. `kind` + `importance` (1-5) feed the relevance x importance x recency blend;
#    `metadata` carries a thread key so retrieval can pull in neighbouring turns.
thread = {"session": "project-x"}
mem.remember("Decision: the primary database is PostgreSQL.", kind="constraint", importance=5, metadata=thread)
mem.remember("We chatted about the weather for a while.", kind="chat", metadata=thread)
mem.remember("The launch date is September 14.", kind="fact", importance=5, metadata=thread)

# 2) Assemble a budgeted, prompt-ready context block (highest-value first, dated, deduped).
print("--- assembled context ---")
print(mem.assemble("When do we launch?", token_budget=128, window=1, thread_key="session"))

# 3) Ranked, source-traceable recall.
print("\n--- ranked recall ---")
for hit in mem.recall("which database did we pick?", limit=3):
    print(f"  {hit.score:.2f}  {hit.record.content}")

# 4) Beliefs revise: a newer fact supersedes the stale one. `memory_state` shows the LIVE value, not both.
mem.remember("The launch date moved to October 1.", kind="fact", importance=5, metadata=thread)
print("\n--- live state (durable beliefs) ---")
for record in memory_state(mem):
    print(f"  {record.content}")

# 5) The guard: memory can inform planning, but acting in the world needs a user-CONFIRMED memory.
confirmed = mem.remember("User confirmed: deploy to staging is approved.", kind="constraint",
                         provenance="user_confirmation")
fact = [hit.record for hit in mem.recall("which database did we pick?", limit=1)]
print("\n--- guard (act only on confirmed memory) ---")
print("  external_action on a confirmed memory:", decide_memory_use([confirmed], intended_use="external_action").allowed)
print("  external_action on a plain fact:      ", decide_memory_use(fact, intended_use="external_action").allowed)
