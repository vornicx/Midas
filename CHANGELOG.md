# Changelog

Notable changes to Midas. Pre-1.0 — the API may change. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- **Core SDK** — `Memory` (`remember` / `recall` / `build_context` / `assemble`) ranking by
  relevance × importance × recency, with same-thread neighbour-window expansion and budgeted,
  highest-value-first context assembly. No LLM at ingest or query.
- **Embedders** — `HashingEmbedder` (offline, zero-dep), `LocalEmbedder` (fastembed/ONNX, bge-base),
  `OpenAIEmbedder`, and `DiskCachedEmbedder` (persistent SQLite cache keyed by model + dim + text).
  `LocalReranker` (cross-encoder, length-capped to avoid ONNX crashes).
- **Stores** — `InMemoryStore` with a vectorised cosine scan over a **cached** embedding matrix
  (numpy; comfortable to ~1M memories) and an identical pure-Python fallback; `SQLiteStore` for
  **persistence across restarts** with **no native extension** (pure stdlib sqlite3).
- **Hybrid retrieval** (BM25 fused with semantic) — off by default; see `BENCHMARKS.md` for the
  honest negative result on conversational data.
- **Belief revision** (supersession) for typed durable facts — off by default; chat never supersedes
  chat. Paraphrased updates are caught by the embedder's cosine similarity (no hand-tuned synonym map),
  so it generalizes beyond any one dataset.
- **Local NLI** (`midas/nli.py`, LLM-free) — a small int8 ONNX MNLI cross-encoder (onnxruntime +
  tokenizers, ~70 MB, no torch/API). Powers **contradiction-gated conversational belief revision**:
  a chat turn revises an earlier belief only when NLI scores it an actual contradiction. This *fixes*
  the cue-heuristic's over-supersession (LongMemEval temporal recall restored 0.76 → **0.95**) while
  staying precise on real updates — closing the "cheap no-LLM contradiction detection" open problem.
  Also exposes **post-hoc answer-grounding** (`--answer-verify-nli`) — override to "I don't know" when
  no retrieved turn entails the answer. Honest result: it does NOT reliably improve abstention (a
  deterministic-reader A/B is unchanged, 0.37→0.37) because the confabulation is drawn from a retrieved
  distractor that *entails* it. Abstention/Calibrated remains the open frontier; see docs.
- **Time-aware retrieval** (LLM-free) — memories carry real **event time** (`remember(created_at=…)`);
  `recall`/`build_context` take a query `now` so recency decays from when a question is asked, context
  renders true dates (UTC), and a "today" header anchors relative-time reasoning. Bitemporal signal,
  no LLM. Eval ablation: `--midas-no-time`.
- **Selective forgetting + temporal tiers** (LLM-free) — `Memory.forget_decayed()` evicts the
  lowest-value memories (`memory_value` = importance × recency) to bound storage and context growth,
  **protecting the durable tier** (facts/preferences/constraints, high importance) and never orphaning
  a supersession chain; returns the forgotten ids (deletion audit trail). `Memory.tier()` names a
  memory's horizon — short (≤1d) / medium (≤1w) / long (multi-day). Measured with `eval/retention.py`
  (eviction policies at the same retained budget): on data with an importance signal, value-based
  forgetting **holds recall@k 1.00 at 25–50% retention** while recency/random eviction fall to
  0.17–0.60; on uniform-importance chat it **reduces to recency** (honest — needs a per-turn importance
  signal, the next step) while cutting context tokens ~3×. Purely additive: no-forget recall@k
  unchanged (LoCoMo 0.62).
- **Content importance scoring** (`ContentImportance`, LLM-free) — derive a turn's importance 1–5 from
  content alone (content-word density, numbers/dates, proper nouns, anti-backchannel); `Memory(
  importance_scorer=…)` auto-applies it to turns ingested without one, so raw chat gets a salience for
  forgetting/tiering. Measured: as a forgetting **protection** it lifts LoCoMo recall@k under eviction
  from 0.10 (recency) to **0.18** (sheds filler, keeps facts); as a pure rank it helps only at moderate
  compression. Honest next lever: novelty-vs-store salience.
- **Extractive consolidation** (`Memory.consolidate`, LLM-free) — collapse near-**duplicate** restatements
  to the single highest-value copy (cosine ≥ threshold, chains preserved); extractive (drops redundant
  records, keeps provenance — never LLM-rewrites). Measured safe (recall@k held: LoCoMo 0.27→0.26 dropping
  10 dups at 0.92); yield is modest at safe thresholds on paraphrase-heavy data and grows with literal
  redundancy/scale.
- **MCP server** (`python -m midas.mcp_server`) — `remember` (auto-derives importance from content),
  `recall` (source-traceable), `build_context`, `maintain` (no-LLM retention: dedup + selective
  forgetting, returns the **deletion audit** of removed ids), `stats` (counts + temporal-tier
  distribution), `forget`, `forget_all`. Optional SQLite persistence via `MIDAS_MCP_DB`; optional
  **bounded memory** via `MIDAS_MCP_MAX_RECORDS` (auto-forget the lowest-value tail over the cap). The
  privacy/cost/provenance/retention surface for long-running and enterprise agents.
- **Zero-config auto-memory** (LLM-free) — install the MCP server and Midas starts remembering on its
  own. The server **injects a memory policy** into the agent (MCP `instructions` + a `memory_session`
  prompt): recall-then-`capture`. `Memory.capture()` + `MemoryPolicy` impose the relevance parameters —
  it scores each turn's importance, enforces a floor (`MIDAS_MCP_MIN_IMPORTANCE`, default 2) and skips
  duplicates, and reports stored/skipped + why. The agent captures freely; Midas decides what's kept.
- **Eval harness** (`eval/`, dev-only) — LoCoMo + LongMemEval loaders, deterministic `recall@k`,
  per-adapter cost/latency instrumentation, and an optional LLM judge (hosted or local Ollama,
  seed-pinned + serialized for reproducibility). **Reader and judge models are decoupled**
  (`--reader-model` vs `--judge-model`) so correctness can be measured with a fixed judge while
  sweeping readers — the apples-to-apples protocol published leaderboards use (e.g. gpt-4o judge).
- **Artifacts** — `BENCHMARKS.md` (reader-independent results + reproduce commands),
  `docs/research-notes.md` (measured findings), a coding-agent demo, PEP 561 typing (`py.typed`),
  and an MIT license.

### Measured (see BENCHMARKS.md)
- Retrieval `recall@k`: LongMemEval-`s` **0.95** (n=40, time-aware) and LoCoMo **0.85** (5
  conversations) vs a recency-window baseline ≤0.03. Time-awareness lifts **temporal recall@k
  0.86→0.95** (deterministic A/B, `--midas-no-time`), no real regression elsewhere.
- Answer correctness (reader = gpt-4.1-mini, n=40): Midas **0.82** vs baseline **0.05**. Per-category
  answer deltas are within run-to-run judge noise at n≤13, so we lead with `recall@k`.
- **Same-reader head-to-head (judge=gpt-4o, structured answerer):** Midas **0.84 @ gpt-4o = SOTA
  Observational Memory's 0.84**, with **zero LLM at ingest** (OM runs an LLM per conversation);
  0.87–0.89 @ gpt-5-mini vs OM 0.95. Midas leads multi-session (0.89 vs 0.872).
- Structured answerer (extract relevant dated entries + compute time deltas before answering) lifts
  non-reasoning readers (gpt-4o 0.76→0.84) and is neutral for reasoning readers.
- Ingest cost: **0 LLM calls, $0 API, 0 data egress** (local embeddings only).
- In-memory recall latency ~0.2 µs/record after matrix caching (~70× the naive Python scan).

### Notes
- Reader-independent metrics (`recall@k`, cost) are primary; end-to-end answer correctness is
  reader-dominated and reported as secondary/noisy — see `docs/research-notes.md`.
