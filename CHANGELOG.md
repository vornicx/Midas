# Changelog

Notable changes to Midas. Pre-1.0 — the API may change. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- **BEAM adapter — the 10M-token frontier benchmark** (`eval.runner --dataset beam --beam-tier
  100K|500K|1M|10M`). BEAM ("Beyond a Million Tokens", ICLR 2026) is the regime where
  context-stuffing is physically impossible; its questions carry `source_chat_ids` evidence, so
  Midas's deterministic reader-independent `recall@k` applies on the frontier benchmark. Loader
  maps real `time_anchor` event times (bitemporal signal) and BEAM's abstention category onto the
  runner's unanswerable semantics; rubric-only categories contribute retrieval metrics only.
  Landscape research + positioning in [`docs/frontier-2026.md`](docs/frontier-2026.md).
- **Bitemporal belief history** — retired beliefs now carry a validity bound (`superseded_at` =
  the revising record's event time) and `recall(as_of=…)` (MCP: `recall(as_of="YYYY-MM-DD")`)
  answers "what did memory say on date X": later records are excluded and supersession chains
  resolve to the version valid then. Zep-class historical queries with no graph DB and no LLM.
- **Background auto-maintain (sleep-time at $0)** — `MIDAS_MCP_AUTO_MAINTAIN=<minutes>` runs a
  periodic no-LLM upkeep pass (consolidate near-duplicates + re-bound the store) while the agent
  is idle — the sleep-time-compute insight without an LLM rewriting memory.
- **Pinned standing directives (default on in MCP)** — durable user rules ("from now on…",
  "always…") are detected at ingest (cue regex, no LLM, user-voiced turns only —
  `is_standing_instruction`, `Memory(detect_standing=True)`) and `build_context(pinned_limit=…)` /
  `MIDAS_MCP_PINNED` (default 2) pins them into every context regardless of query relevance: the
  no-LLM version of Letta's always-in-context core memory. Measured on BEAM-100K:
  instruction-following recall@k **0.26 → 0.44** (pinned=2) **→ 0.51** (pinned=4), overall
  0.56 → **0.58**. The first detector iteration (which also pinned assistant-voiced advice) hurt
  across the board and is documented as the negative that set the user-voice-only rule.

### Measured negative (kept opt-in, default off)
- **Hit-anchored expansion / pseudo-relevance feedback** (`recall(anchor_boost=…)`,
  `--midas-anchor-boost`) — letting records near a confirmed top hit earn `boost × cosine`
  relevance was the natural attack on the multi-evidence categories (event-ordering 0.24 on
  BEAM-100K), and it *hurts monotonically*: overall recall@k 0.56 → 0.44 (boost 0.7) → 0.28
  (0.85), with the target category itself dropping to 0.17. Anchor-similar records are
  conversational near-duplicates, not gold. Third member of the same measured family (hybrid
  fusion, thread-cap): **reshuffling bge's dense ranking only ever degraded it** — the remaining
  aggregation-category gap is a reader/structure problem, not a retrieval-reordering one.

### Fixed
- **Belief revision did not survive restarts on SQLite-backed stores** — supersession mutated the
  in-memory mirror without persisting (`store.put`), so `superseded_by` silently vanished on
  reopen. Found while adding the bitemporal stamp; regression test included.

### Added (earlier this cycle)
- **TypeScript port (experimental)** — `packages/midas-ts`, `npx -y midas-memory-mcp`. Same MCP
  tool surface, env knobs, injected policy, **and SQLite schema** as the Python server; the
  hashing embedder is bit-comparable (md5 parity pinned by a Python-generated fixture), so a TS
  and a Python server can **share one DB file live** (both probe `data_version`) — verified
  bidirectionally in tests (13 node:test cases). Ships the core: ranking + parsimony floor,
  BM25+RRF hybrid (cached), supersession chains, policy-gated capture, structural importance,
  guard, lean `build_context`, selective forgetting, `forget_matching`. Not ported yet: ONNX
  semantic embeddings, NLI, reranker — the Python server stays the reference.
- **Multilingual embeddings** — `MIDAS_MCP_EMBEDDER=multilingual` (or any fastembed model id)
  selects `paraphrase-multilingual-MiniLM-L12-v2`; runner gains `--local-model` and a
  `synthetic-es` dataset. Measured both ways: on Spanish content the English-only bge-base drops
  to **answer_dumb 0.68** while the multilingual model scores **1.00** (both 1.00 on the English
  twin); on English LongMemEval-`s` (n=40) the multilingual model retrieves at **0.83 vs bge's
  0.95** — switch only for non-English memory. The English-trained cross-encoder reranker stays
  off in multilingual mode.
- **Opt-in ANN for big stores** — `InMemoryStore(ann_threshold=…)` / `SQLiteStore(…)` /
  `MIDAS_MCP_ANN=1` route search through the numpy-only IVF index at ≥10k records (index cached on
  the store's change counter; predicate pushdown; approximate — recall ~0.95 at nprobe=16 per
  BENCHMARKS §4, which is why exact scan stays the default).

### Fixed
- **`LocalEmbedder` hardcoded `dim=768`** — any non-default model (e.g. 384-d MiniLM) silently
  poisoned the on-disk embedding cache with mislabelled rows that failed to decode in later
  sessions. The dim now comes from the model registry (probe fallback), and `DiskCachedEmbedder`
  refuses to write a vector that contradicts the declared dim (regression test).

### Measured negative (kept opt-in)
- **Thread-diversified recall** (`recall(thread_cap=…)`, `--midas-thread-cap`) — capping hits per
  session-thread to help multi-evidence questions *hurts* across the board on full LongMemEval-`s`
  (overall recall@k 0.92 → 0.84 at cap=3, 0.89 at cap=5): gold evidence is often consecutive turns
  of one session, and the cap evicts it for other-session distractors. Trace analysis showed the
  remaining temporal misses are multi-evidence spread (not query-date parsing), so no date-window
  heuristic ships either.

### Measured
- **BEAM, all four tiers complete (100K → 10M tokens), deterministic.** Midas recall@k
  **0.56 → 0.51 → 0.40 → 0.32** across a 100× scale-up while the recency baseline scores **0.00 at
  every tier**; knowledge-update holds at 0.68 even at 10M. The 10M tier (208,696 turns per the 10
  conversations) ingested in hours of local CPU at $0 — the cost regime where LLM-at-ingest
  systems pay for ten million tokens of extraction per conversation. Honest weak side documented:
  aggregation abilities collapse at 10M (instruction 0.00, event-ordering 0.02, summarization
  0.03) — whole-conversation abilities top-k retrieval cannot cover by construction.
- **Full-set retrieval headlines — no sampling caveat left.** LongMemEval-`s`, all **500**
  questions (246,750 turns ingested, deterministic, seed 0): Midas recall@k **0.92** vs
  recency-baseline **0.01**; per-category fact 0.97 (n=126) · knowledge-update 0.93 (n=78) ·
  temporal 0.91 (n=133) · multi-session 0.89 (n=133) · preference 0.89 (n=30). Together with the
  full LoCoMo set (0.73, n=1,540) both retrieval headlines now run on complete public question
  sets. Judged answer-correctness remains n=40 (hosted-reader cost-bound).

### Fixed
- **LoCoMo benchmark corrected to the full public set: recall@k 0.73 vs baseline 0.05 (n=1,540).**
  The previously published **0.85 (n=50) did not reproduce** against the publicly downloadable
  `locomo10.json` (verified not to be code drift: the v0.0.1 harness gives the same 0.28 on that
  sample today). Root cause of the full-set gap: the runner's old LoCoMo-specific
  `min_relevance=0.75` absolute floor — tuned on the early sample — pruned most gold turns at
  scale (recall@k **0.18 with it, 0.73 without**). The floor is removed; the SDK's scale-free
  `min_relevance_ratio` is the safe replacement. BENCHMARKS.md carries the correction notice.

### Added
- **Scale-free context parsimony, default on** (`Memory(min_relevance_ratio=0.3)`; per-call
  override; `0` disables; runner flag `--midas-min-relevance-ratio`) — recall drops any hit whose
  relevance is below 0.3× the query's own top hit. Measured (deterministic): **zero gold evicted on
  any dataset**, ~**2× precision@k**, ~**30–40% fewer context tokens** on spread-scale embedders
  (synthetic 102→87, conflicts 207→121, multiday 174→126 avg tokens), and a verified **no-op on
  bge-base** (LongMemEval-`s` recall@k 0.95 and tokens identical). Honest boundary documented:
  0.4+ evicts multiday's buried update (recall 1.00→0.80), so the default stays at 0.3. Unlike the
  absolute `min_relevance` floor, the ratio transfers across embedders because it is relative to
  each query's best hit.

## [0.0.3] — 2026-06-10

Consolidates everything since 0.0.1 (0.0.1 and 0.0.2 were cut without per-release sections).

### Added
- **Live multi-process memory sharing** — `SQLiteStore` now detects writes from *other* connections
  (SQLite `PRAGMA data_version`) and refreshes its in-memory mirror, so several MCP clients
  (Claude Code + Claude Desktop + an IDE) pointed at one DB file see each other's captures live,
  without restarts. The shared connection is also lock-guarded and usable from worker threads (how
  MCP frameworks actually run sync tools).
- **Namespaces (scoped memory)** — share one DB across projects/agents/users without cross-talk.
  SDK: `recall`/`build_context`/`forget_matching` take `metadata_filter={...}` (equality scoping;
  neighbour-window expansion respects it too — no leaks past the filter). MCP: `MIDAS_MCP_NAMESPACE`
  env sets the server's default scope; every tool also takes a per-call `namespace`; `stats` reports
  a `by_namespace` breakdown. Unscoped behaviour is unchanged.
- **Topic-level erasure with audit** (`Memory.forget_matching`, MCP `forget_matching`) — "forget what
  you know about X": matches by relevance, **dry-run by default** in MCP (preview what would be
  deleted, then confirm with `dry_run=false`), returns the full list of removed records as the
  erasure audit trail. Deliberately bypasses durability protections — an explicit erasure request
  outranks retention. The right-to-be-forgotten lever, no LLM.
- **Chain-safe single deletion** (`Memory.forget`, now used by the MCP `forget` tool) — deleting a
  record mid-supersession-chain relinks the chain instead of orphaning it, so a query phrased like
  the old value still resolves to the current belief.
- **MCP `build_context` upgrades** — the context block now carries the measured temporal grounding
  ("# Today is …" header + per-memory relative ages — the LLM-free signal that lifted temporal
  recall@k 0.86→0.95 in the eval) and exposes `limit`, `hybrid`, and `namespace`.

- **Token-lean by default** — the context an agent actually pays for is now compact. The injected
  MCP policy text shrank **442 → 198 approx tokens (−55%)** (kind/provenance taxonomies live in the
  `remember`/`capture` tool descriptions instead of being repeated); `build_context` emits lean
  memory lines by default (`- [kind | date] …`, **−42%/line** vs the audit format —
  `Memory(include_provenance=False)` is the new default, with a per-call `include_provenance`
  override), and the budget accounting now charges the `[source: …]` suffix it previously appended
  for free. Full provenance/source evidence stays one `recall`/`inspect_memory` call away; eval
  adapters already ran lean, so benchmark numbers are unchanged.
- **Hybrid recall is ~8× cheaper on a stable store** — the BM25 index is cached on the store's
  change counter (rebuilt only after writes) and scores via per-term posting lists, so a query
  touches only documents sharing a term with it. Stable 5k-record store: ~66 ms → **~8 ms/query**;
  identical scores and eval metrics (conflicts-v1 and LongMemEval-`s` reproduce exactly).
- **Measured negative (hybrid stays opt-in)** — on LongMemEval-`s` (n=40, bge-base, deterministic)
  hybrid BM25+RRF *hurts* retrieval: multi-session recall@k 0.97→0.81, temporal 0.95→0.86
  (fact 0.89→0.90, within noise). Lexical rank-fusion displaces buried semantic evidence on
  paraphrased queries. Hybrid therefore stays **off by default** and is recommended only for
  exact-identifier lookups (error codes, ticket ids, function names), where BM25 catches what the
  bi-encoder ranks low.
- **Eval methodology doc** (`docs/methodology.md`) — anti-cheating checklist (no query rewriting, no
  LLM at ingest, seeded sampling), verbatim MCP-injected policy, how conflicting memories are handled
  (supersession chains + gating), dumb-reader ablation, conflicts-v1 results, and publishable failure
  traces for selective forgetting.
- **Dumb-reader ablation** (`--dumb-reader` on `eval.runner`) — deterministic extractive reader (no
  LLM); adds `answer_dumb` to the leaderboard. If it tracks `recall@k`, headline numbers are not
  reader-inflated.
- **`conflicts-v1` benchmark** — adversarial near-duplicates + temporal conflicts
  (`eval/datasets.conflicts`, `eval.multiday --dataset conflicts`, `eval.runner --dataset conflicts`).
  Reports `ctx_stale` / `ctx_contradict` on the main leaderboard when the dataset annotates outdated
  values.
- **Retention forgetting traces** (`eval.retention --trace`) — per-question audit after eviction
  (gold survived vs evicted, value-vs-fifo wins and failures).
- **Guard / provenance control-plane** — `Armorer` + `Guard` (`midas/guard.py`); four provenance tags;
  MCP `check_memory_use`; mixed-recall bundles allow actions when at least one hit satisfies policy
  (invalid hits reported in `blocked_ids`, not a whole-decision veto).
- **MCP server distribution** — new `midas-memory-mcp` launcher on PyPI, listed on the **official
  MCP registry** (`io.github.vornicx/midas`); run install-free with `uvx midas-memory-mcp`. The MCP
  server now reports its own version in the handshake (previously the MCP SDK version).
- **Core SDK** — `Memory` (`remember` / `recall` / `build_context` / `assemble`) ranking by
  relevance × importance × recency, with same-thread neighbour-window expansion and budgeted,
  highest-value-first context assembly. No LLM at ingest or query.
- **Embedders** — `HashingEmbedder` (offline, zero-dep), `LocalEmbedder` (fastembed/ONNX, bge-base),
  `OpenAIEmbedder`, and `DiskCachedEmbedder` (persistent SQLite cache keyed by model + dim + text).
  `LocalReranker` (cross-encoder, length-capped to avoid ONNX crashes).
- **Stores** — `InMemoryStore` with a vectorised cosine scan over a **cached** embedding matrix
  (numpy; comfortable to ~1M memories) and an identical pure-Python fallback; `SQLiteStore` for
  **persistence across restarts** with **no native extension** (pure stdlib sqlite3).
- **float32 in-memory embeddings** — records store the embedding as a float32 numpy array, not a
  Python `list[float]` (~32 B/value). Measured ~**7× smaller footprint** at 768 dims (a 1M-record
  in-memory store drops from ~24 GB to ~3.5 GB) and **faster queries** (float32 matmul); SQLite already
  persisted float32. Measured by `eval/bench_perf.py` (latency / throughput / real tracemalloc footprint
  — the numbers the project had never measured).
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
- **Entity-grounded abstention** (`midas/entity.py`, LLM-free) — a new abstention lever orthogonal to
  cosine/NLI: abstain when the answer's source turn is about a *different entity* than the question asks
  (the diagnosed confab-from-distractor root cause). Dropping recurring *attribute* words makes the focus
  the entity noun; **offline-validated 8/8** on the diagnosed failure cases (incl. "favorite colour" vs
  "favorite food", "city" vs "Barcelona"), 11 tests. Honest limit: crafted cases — the end-to-end win
  needs a capable reader (local 1B doesn't confabulate-from-distractor; hosted credits exhausted).
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
  compression.
- **Novelty-vs-store importance** (`Memory(novelty_weight=…)`, LLM-free) — blends importance with
  `1 − max-cosine-to-store` so a *new* fact can outrank a *repeated* one. **Off by default: a measured
  negative.** At equal budget it is neutral on LoCoMo/synthetic recall@k and *harmful* on multiday
  (1.00 → 0.60), because repetition usually signals importance and demoting restated gold evicts it.
  Kept as a tested, opt-in knob; its right home is consolidation (dedup), not eviction-ranking.
- **Reinforcement importance** (`Memory(reinforce=True)`, LLM-free) — the inverse of novelty: a restated
  turn *boosts* the matched memory's importance + recency (repetition ⇒ salience); in `capture` a
  restatement reinforces the existing memory and is skipped. **Off by default: also a measured negative**
  — recall@k drops at equal budget (LoCoMo 0.08→0.03 @25%; multiday 0.60→0.40 @25%). Unifying finding:
  on raw conversation **repetition tracks commonness, not importance**, so neither novelty nor
  reinforcement improves no-LLM forgetting. Content-salience as a *protection* stays the best signal.
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
- **Eval harness** (`eval/`, dev-only) — LoCoMo + LongMemEval + **multiday** + **conflicts-v1**
  loaders; deterministic `recall@k` / `precision@k`; optional **`answer_dumb`** (`--dumb-reader`);
  optional **`ctx_stale` / `ctx_contradict`**; per-adapter cost/latency instrumentation; optional LLM
  judge (hosted or local Ollama, seed-pinned + serialized for reproducibility). **Reader and judge
  models are decoupled** (`--reader-model` vs `--judge-model`). Retention harness with **`--trace`**
  forgetting audits. See [`docs/methodology.md`](docs/methodology.md).
- **Artifacts** — `BENCHMARKS.md` (reader-independent results + reproduce commands),
  `docs/methodology.md` (anti-cheating + failure cases + verbatim policy),
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
- Reader-independent metrics (`recall@k`, `precision@k`, `answer_dumb`, cost) are primary; end-to-end
  answer correctness is reader-dominated and reported as secondary/noisy — see
  [`docs/methodology.md`](docs/methodology.md) and `docs/research-notes.md`.
