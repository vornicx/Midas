# The agent-memory frontier (mid-2026) — landscape, positioning, roadmap

Research notes from a structured sweep of the frontier systems' public docs, papers, and
benchmarks (June 2026). Each claim links to its source. This document drives what Midas adopts,
what it deliberately rejects, and what it measures next.

## 1. The strategic shift: the 10M-token regime

The field's own leaders now argue that **LoCoMo and LongMemEval are saturated**: million-token
context windows let a naive "dump everything into context" approach score competitively, so those
benchmarks "can no longer distinguish a real memory system from a context stuffer"
([Hindsight, Apr 2026](https://hindsight.vectorize.io/blog/2026/04/02/beam-sota)). The successor is
**BEAM** ("[Beyond a Million Tokens](https://arxiv.org/abs/2510.27246)", ICLR 2026): 100 coherent
conversations at **128K/500K/1M/10M tokens**, 2,000 validated questions across **ten memory
abilities** — including contradiction resolution, event ordering, and abstention. At the 10M tier
*no context window fits*; only a real memory architecture works.

**Why this regime favours Midas structurally.** Every LLM-at-ingest system pays tokens
proportional to corpus size: at 10M tokens per conversation, extraction passes cost dollars *per
conversation* and re-processing compounds. Midas ingests at local-embedding prices (~cents of CPU,
$0 API, zero egress) at any scale, and BEAM's questions carry `source_chat_ids` evidence — so our
deterministic, reader-independent `recall@k` applies on the frontier benchmark
(`eval.runner --dataset beam`).

## 2. The frontier systems, and what Midas takes from each

| System | Core idea | What Midas adopts (no-LLM translation) |
|---|---|---|
| **[Zep / Graphiti](https://github.com/getzep/graphiti)** ([paper](https://arxiv.org/abs/2501.13956)) | Temporal context graphs: facts as edges with **validity windows** ("when it became true, when it was superseded"), episodes as provenance | **Bitemporal validity windows on supersession chains**: every retired belief stamps `superseded_at`; `recall(as_of=…)` answers "what did we believe on date X" — no graph DB, no LLM |
| **[Letta](https://www.letta.com/blog/sleep-time-compute)** ([paper](https://arxiv.org/abs/2504.13171)) | Sleep-time compute: a second agent reorganises memory during idle time | **`MIDAS_MCP_AUTO_MAINTAIN`**: periodic background consolidate + re-bound pass — the idle-time upkeep insight at $0 (extractive, not LLM rewrites) |
| **[Mem0](https://mem0.ai/research)** | Single-pass hierarchical LLM extraction + multi-signal retrieval; LoCoMo 92.5 / LongMemEval 94.4 / BEAM-1M 64.1 (judged, ~7k tokens/call) | The *benchmark discipline* (BEAM adapter); extraction itself is rejected (below) |
| **[Hindsight](https://hindsight.vectorize.io/blog/2026/04/02/beam-sota)** | LLM retain/reflect pipeline; #1 on BEAM-10M | The framing that 10M is where architecture shows; their cost structure is the foil |
| **[LIGHT](https://mohammadtavakoli78.github.io/beam-light/)** (BEAM authors) | Episodic memory + working memory + LLM scratchpad (+3.5–12.7% on BEAM) | Episodic retrieval + recency window ≈ what `recall` + session-window expansion already do; the scratchpad is LLM-bound (rejected) |
| **[Mastra OM](https://mastra.ai/blog/observational-memory)** | LLM Observer/Reflector compaction; LongMemEval 94.9 @ gpt-5-mini | Already measured against (Midas ties at gpt-4o with $0 ingest; BENCHMARKS §6) |
| **[LangMem](https://langchain-ai.github.io/langmem/)** | Hot-path memory tools + background LLM memory manager | The hot-path/background split (capture vs auto-maintain) — both no-LLM in Midas |

## 3. What Midas deliberately does NOT adopt

**LLM extraction/summarization at ingest** — the entire frontier pack (Mem0, Hindsight, OM, LIGHT's
scratchpad, LangMem's manager) buys its judged-answer headroom with per-token ingest cost, latency,
egress, and an extraction step that can silently hallucinate. Midas's bet stays: verbatim,
source-traceable records + retrieval quality + reader-side structure, at $0 marginal ingest. The
honest consequence: **summarization-type questions are a structural weakness** (we return turns,
not abstracts) — visible in BEAM's per-category results and stated rather than hidden.

## 4. Measured so far (deterministic, reproducible)

See BENCHMARKS §"BEAM" for current numbers (`recall@k`/`precision@k` vs a recency baseline, plus
`answer_dumb` where BEAM provides reference answers; instruction_following / preference_following /
summarization are rubric-graded upstream → retrieval metrics only; abstention maps to the
unanswerable category).

## 5. Roadmap (eval-first, in order)

1. ~~BEAM 500K → 1M → 10M tiers~~ — **done, all four tiers** (recall@k 0.56/0.51/0.40/0.32 vs a
   0.00 baseline everywhere; BENCHMARKS §BEAM). The cost story held: the 10M tier ingested for
   hours of local CPU at $0.
2. **Judged BEAM runs** with a fixed hosted reader (pending API credits) for cross-system
   answer-rate comparison vs the Mem0/Hindsight published numbers.
3. **Bitemporal recall on BEAM event-ordering/temporal categories** — measure whether
   `as_of` + validity windows lift the time-sliced questions.
4. **TS port parity**: local ONNX semantic embeddings in `packages/midas-ts`.
5. Entity index (`midas/entity.py` exists) as a no-LLM nod to graph memory — only if a measured
   win shows on multi-hop categories.

*Method note: sources were collected with the context.dev scrape API (clean-markdown captures of
each system's public docs on 2026-06-12) and are quoted from those captures.*
