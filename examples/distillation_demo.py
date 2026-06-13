"""Demo: why DISTILLATION beats dumping raw turns — measured, no LLM inside Midas.

The frontier's extra answer-quality (Mem0/Letta/LIGHT) comes from distilling raw conversation into
compact, self-contained facts at ingest — they pay a separate LLM for it. Midas's agent-driven tier
gets the same effect for $0: the LLM the agent is *already running* writes one compact fact, and
Midas (which never calls an LLM) indexes it.

This demo makes the mechanism visible. Same conversation, same chatter in both stores — the only
difference is whether the durable decision was stored as a **raw turn** or as a **distilled fact**.
A later, differently-phrased query then shows the distilled fact ranking first by a clear margin,
while the raw turn gets buried under irrelevant chatter.

Run:  uv run --no-sync python examples/distillation_demo.py
      Install the `local` extra (`pip install -e ".[local]"`) for real semantic embeddings;
      otherwise it falls back to the offline hashing embedder (the gap is smaller but still visible).
"""
from __future__ import annotations

import sys

from midas import HashingEmbedder, Memory, StructuralImportance


def build_embedder():
    try:
        from midas import LocalEmbedder

        return LocalEmbedder(), "local semantic (bge-base)"
    except Exception as exc:  # fastembed not installed
        print(f"(local embedder unavailable: {exc}; using offline hashing)\n", file=sys.stderr)
        return HashingEmbedder(), "offline hashing (lexical)"


# A realistic session: one durable decision, said the way people actually say it, buried in chatter.
CHATTER = [
    "user: morning! how was your weekend",
    "user: lol yeah the weather's been wild",
    "user: ok so I need to sort out a few things on the project today",
    "user: btw did you see the new framework release? looks interesting",
    "user: anyway back to work haha",
    "user: oh and remind me to grab coffee later",
    "user: the standup ran long again today ugh",
    "user: someone broke the lint check, classic",
]
RAW_DECISION_TURN = (
    "user: you know what, for now let's just point everything at staging, prod isn't ready yet"
)
# What the agent's own LLM would write in the agent-driven tier (Midas still calls no LLM):
DISTILLED_FACT = (
    "Deployment target is staging; production is deferred until it is ready (decided this session)."
)
QUERY = "which environment do we deploy to?"


def main() -> None:
    embedder, label = build_embedder()
    print(f"embedder: {label}\nquery:    {QUERY!r}\n")

    # --- Store A: dump raw turns (no distillation) ---
    raw = Memory(embedder=embedder, importance_scorer=StructuralImportance())
    raw.remember_many([{"content": t, "kind": "chat"} for t in CHATTER])
    raw.remember(RAW_DECISION_TURN, kind="chat")

    # --- Store B: the agent distilled the session into one compact fact ---
    distilled = Memory(embedder=embedder, importance_scorer=StructuralImportance())
    distilled.remember_many([{"content": t, "kind": "chat"} for t in CHATTER])
    distilled.remember(DISTILLED_FACT, kind="constraint", importance=5)

    for name, mem in [("RAW TURNS (dump everything)", raw), ("DISTILLED (agent wrote one fact)", distilled)]:
        print(f"=== {name} — top 3 ===")
        for rank, hit in enumerate(mem.recall(QUERY, limit=3), 1):
            star = "  <- the deploy answer" if "staging" in hit.record.content.lower() else ""
            print(f"  {rank}. score {hit.score:.2f}  {hit.record.content[:68]}{star}")
        print()

    print(
        "Takeaway: a single CLEAN fact retrieves better than its conversational fragment — cleaner\n"
        "relevance plus typed importance, and the agent's own LLM wrote it, so Midas stayed no-LLM.\n"
        "\n"
        "HONEST CAVEAT (measured): this intuition does NOT generalise to compressing a WHOLE evolving\n"
        "conversation. On a judged BEAM A/B, naive distillation HURT (replacing raw turns: answer\n"
        "0.30 -> 0.08) because lossy summaries drop the temporal/changed-value detail. The frontier's\n"
        "lift needs sophisticated, structure-preserving extraction (key-value + time), not a generic\n"
        "summary — so the distill dial is off by default, keep_raw when on. See BENCHMARKS.md."
    )


if __name__ == "__main__":
    main()
