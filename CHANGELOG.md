# Changelog

Notable changes to Midas. From 1.0.0 the public surface — the `midas.Memory` SDK (remember / recall /
consolidate / supersession / forget), the guard (`guard_reliance`, `decide_memory_use`, the coding
`is_forbidden` gate), the MCP tools, and the `midas` CLI — follows [semver](https://semver.org):
breaking changes only land in a major. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [1.0.0] — 2026-07-06

The 1.0 contract: the public surface — the `midas.Memory` SDK, the guard, the coding `is_forbidden`
gate, the MCP tools, and the `midas` CLI — is now locked under semver. This release consolidates and
hardens everything since 0.1.1 (the retrieval core, the governance guard, the control-plane, the audit
chain, the client tooling, and the TypeScript port) and freezes the API.

### Added
- **The continuity control-plane** (`midas/continuity.py` + MCP tools): **`resume`** — everything an
  agent needs to pick up a session in ONE call (pinned directives, forbidden rules, current state,
  what changed, open loops, unresolved conflicts; token-budgeted, prompt-ready); **`memory_conflicts`**
  — live beliefs that contradict each other with neither superseding the other (the multi-agent
  shared-memory failure mode), NLI-scored when available with a same-slot heuristic fallback, ranked
  candidates only; **open loops** — `remember_commitment` / `open_loops` / `close_loop` make promised
  work survive across sessions with an auditable promise→resolution chain. The injected agent policy
  now starts sessions with `resume` and keeps promises via open loops.
- **Per-kind TTL retention**: `Memory.forget_expired({"chat": 30})`, `MIDAS_MCP_TTL="chat=30,note=90"`,
  and `maintain(ttl=…)` — age-based retention next to value-based forgetting; user-confirmed,
  standing, and supersession-chain records never expire silently.
- **Tamper-evident audit chain** in `SQLiteStore`: every put/delete/clear appends a hash-chained
  entry (hashes only — never memory content); `midas audit [--json]` shows and verifies the chain and
  `midas doctor` checks its integrity. `audit=False` opts out for perf-sensitive paths.
- **`midas import --from claude-md | cursorrules | jsonl`**: file-based agent memory (CLAUDE.md,
  .cursorrules, exported JSONL) becomes first-class Midas records — sectioned, deduped, idempotent,
  and governance-safe by default (imported rules are `observation` provenance and cannot authorize
  guarded actions unless you pass `--confirmed`).
- **4 more clients in `midas init`**: VS Code (user `mcp.json`, `servers` key), Gemini CLI, Cline,
  and Zed (`context_servers`) — each written in its own config schema; status/uninstall understand
  them too.
- **HTTP bearer-token auth**: `midas serve --http --token <secret>` / `MIDAS_MCP_TOKEN` gates every
  request on the shared-URL transport (constant-time compare, 401 + `WWW-Authenticate` otherwise).
- **`midas doctor --json`** — the machine-readable diagnosis, same envelope as the wiring receipt.
- **TypeScript port: optional local semantic embeddings.** `LocalEmbedder` (ONNX via the optional
  `@huggingface/transformers`, bge-small by default) behind `MIDAS_MCP_EMBEDDER=local`, with a clean
  announced fallback to the byte-parity hashing embedder; the TS `Memory` API is now async
  (`remember`/`capture`/`recall`/`buildContext`/`forgetMatching` return promises).
- **CI on Windows and macOS** (core + MCP): the client-wiring code is full of per-OS paths that only
  Linux exercised.
- **The Agent Continuity Bench measures the new control-plane**: `resume_fidelity` (the one-call pack
  contains live state / forbidden rules / open loops and never presents superseded values or closed
  loops as live), `conflict_detection`, and `conflict_precision` — all → 1.00, wired into
  `midas bench` with the same all-green verdict rule.
- **Inspector views** for the control-plane: Conflicts (ranked contradiction pairs with per-side
  forget), Open loops (close with a recorded resolution), and Audit log (chain verification + a
  hash-only tail).
- **TypeScript parity for the control-plane and audit chain**: `resume`, `memoryConflicts`
  (heuristic tier), open loops, `forgetExpired`/TTL, and the same hash-chained `audit_log` — the hash
  is canonicalised to integer microseconds so **either runtime verifies chains written by the other**
  (validated bidirectionally).
- **`midas init --claude-hook`**: a Claude Code SessionEnd hook (`midas hook capture-session`) that
  offers each finished session's user turns to `capture` — memory even when the agent never calls the
  tool, with the same no-LLM keep/skip policy, fail-open by construction, removed by
  `midas uninstall`.
- **`midas import --from mem0 | zep`**: migrate Mem0 exports (distilled memories → facts with
  original ids/attribution preserved) and Zep exports (facts/edges → facts, messages → chat turns).
- **Encryption at rest (opt-in)**: `SQLiteStore(key=…)` / `MIDAS_MCP_KEY` with the new `[encrypted]`
  extra (SQLCipher). The file on disk is ciphertext; opening without the key fails; if the key is set
  but the extra is missing Midas **fails closed** instead of silently writing plaintext.
- **`midas init --json` / `midas status --json` — a machine-readable client wiring receipt** (#15).
  One command now yields a compact, pasteable proof of what was wired: per client its config path,
  detected/wired/changed state, backup path, and skip reason; plus the memory DB, scope mode
  (shared / project-scoped / manual-namespace), server command, embedder, and the enforced policy.
  `status --json` parses each client config (JSON `mcpServers` and codex TOML) to report the actual
  command + `MIDAS_*` env every client will launch with, and warns when clients disagree on namespaces.
  The receipt is an audit boundary: config paths and scope/policy only — never memory contents.
  The human output of both commands is unchanged. `midas/cli.py`, `tests/test_cli.py`.
- **`midas bench`** — the full governance / safety / continuity bench suite as one CLI command
  (equivalent to `uv run python -m eval.benches` from a checkout; guides to clone otherwise).
  Deterministic, $0, no datasets, no API key — a skeptic can verify every governance number.
- **Stress tests proving the multi-client promise**: 4 threads writing concurrently to one SQLite
  store with no lost or corrupt writes; two `Memory` instances on one file seeing each other's
  writes without a restart; unicode/emoji/20k-char inputs handled and empty/whitespace rejected;
  recall finding the signal among 1,500 noise records.

### Changed
- **`midas inspect` redesigned end to end.** A real light theme (not an inverted dark one) alongside the
  existing dark brand theme, toggle + system-preference default + persisted choice; a fixed, validated
  categorical color system so every `kind` and `provenance` value keeps the *same* color everywhere in
  the app (Overview bars, Browse tags, Project governance) instead of one monochrome accent for
  everything; a real 30-day activity chart (SVG line + area, hover crosshair/tooltip) and a recency
  (short/medium/long) stacked bar on Overview; grouped, badge-annotated navigation (Memory / Coding /
  Governance) with a command palette (**⌘K**) and a **/** search shortcut; native `confirm()`/`prompt()`
  replaced with in-app modal/toast components; a responsive layout down to phone width with a horizontal
  pill nav. New endpoints backing it: `api_timeseries` (daily capture counts), `api_meta` (version/db/
  embedder/audit-chain identity), and `by_tier` added to `api_overview`. Fixed a real routing bug in the
  process: the old build never listened for `hashchange`, so browser back/forward and direct links to a
  view silently did nothing.
- The TypeScript `Memory` API is async (`remember`/`capture`/`recall`/`buildContext`/
  `forgetMatching` return promises) to support the optional ONNX embedder.
- **A bare `Memory()` now auto-selects real semantic recall.** With the `[local]` extra installed,
  `Memory()` upgrades from the offline hashing embedder to `LocalEmbedder` automatically (override
  with `MIDAS_EMBEDDER=hashing|local`; default `auto`). Fixes the out-of-the-box recall weakness an
  external test hit: the SDK silently used the noisy hashing fallback even when semantic embeddings
  were available.

### Security — the governance moat (memory-safety bench: 10 attacks / 4 benign, ASR 0.00, benign-pass 1.00)
- **Prohibition veto.** A live `forbidden_action` rule recalled for an external/destructive action
  vetoes it — overriding any confirmation in the same evidence, so an attacker can't approve around
  a standing "never do X". `midas/guard.py`.
- **Relevance bar for actions.** Authorizing an external/destructive action requires strongly
  relevant evidence (`GUARD_ACTION_MIN_RELEVANCE=0.25`), so a weakly-matched confirmation for a
  *different* action can't lend its approval. Planning/answers unchanged.
- **Provenance integrity.** A user-confirmed belief can no longer be superseded ("laundered") by
  lower-provenance memory — an attacker's *observation* can't overwrite confirmed truth and ride the
  supersession chain past the guard's currency rule. Genuine user re-confirmations still revise.
- **Cross-namespace isolation.** A confirmation scoped to namespace `alpha` does not authorize the
  same action in `beta`, even on an exact text match — proven by adversarial bench cases wired
  through the guard's recall scope filter.

## [0.1.1] — 2026-06-27

### Fixed
- **Recall no longer returns a lone irrelevant memory with the hashing fallback.** The offline hashing
  embedder now declares an absolute noise floor (a hashed cosine ~0 means no token overlap), so a query
  that matches nothing returns empty instead of the only memory present — without dropping real matches
  (which land well above the floor). Semantic embedders declare 0, so the benchmarked path is unchanged.
  `midas/embeddings.py`, `midas/memory.py`. (Found via external testing of 0.1.0.)
- **`quickstart.py` is a real end-to-end demo** (remember → assemble → recall → supersession → guard),
  with `tests/test_quickstart.py` running it as a subprocess so CI catches any break.

### Changed
- **`midas doctor`** now flags clearly when the SDK is installed but the `[mcp]` server extra is not.

## [0.1.0] — 2026-06-26

### Fixed
- **A prohibition can no longer authorize an action.** A `forbidden_action` rule is stamped as a
  user-confirmed constraint, so the mechanical guard counted it as *authorizing* the very action it
  forbids ("never run destructive migrations" → approving a destructive migration). `decide_memory_use`
  now excludes `code_kind=="forbidden_action"` records from authorizing an external/destructive action
  (they still inform planning/answers, and are enforced separately by `is_forbidden`). `midas/guard.py`,
  `tests/test_coding.py`.
- **Guard no longer authorises actions on stale beliefs.** `decide_memory_use` checked provenance and
  actor but not *currency*, so a **superseded** memory — even one user-confirmed when it was current —
  could still justify an answer or an external/destructive action. It now blocks superseded evidence for
  any non-planning use (planning may still weigh history); the guard verifies currency itself instead of
  trusting recall to have filtered the stale record. `midas/guard.py`, `tests/test_guard.py`.
- **Hybrid recall is no longer O(N)/query at scale (~250× faster).** `_hybrid_candidates` rebuilt the
  allowed-record map by scanning the whole store on every query; the no-filter path now reuses a
  version-cached `_records_by_id()` map (and `recall()` drops the predicate when no scope filter is
  active). Measured on LongMemEval-`s` (246,750 turns): **11.6 s/query → 46 ms/query**, recall
  unchanged (0.92, BM25 hybrid). Affects any `--midas-hybrid` use at scale.

### Added
- **The `midas` CLI — one-command setup.** `midas init` creates one shared store
  (`~/.midas/memory.sqlite3`) and wires up the MCP clients it finds (Claude Code, Codex, Cursor, Claude
  Desktop, Windsurf) so they all share one memory; plus `serve --http` (a shared **MCP URL**), `status`,
  `doctor`, `export` / `import` (JSON backup / move machines), `uninstall`, and `update`. `midas/cli.py`.
- **Memory Inspector (`midas inspect`)** — a local, glass-box web UI over your store: an Overview metrics
  dashboard, Browse/search of verbatim records, belief-history time-travel, project state, a what-changed
  diff, the governance/audit verdict, and forget-with-receipt. Stdlib-only, zero egress. `midas/inspector.py`.
- **Per-project memory scoping** — `MIDAS_MCP_NAMESPACE=auto` (or `midas init --project-scoped`)
  partitions the one shared store by project (git repo / cwd), automatically.
- **Agent-memory governance suite** — Memory-Safety (ASR / benign-pass) and Coding benches join Continuity
  (`eval/benches.py`); a coding-agent vocabulary (`remember_code` / `project_state` / forbidden-action
  checks), an audit trail (`audit_use` / content-hashed `forgetting_receipt`), RBAC (`access.py`), and
  live state views (`memory_state` / `memory_diff`).
- **Agent Continuity Bench v0** (`eval/continuity.py`) — a deterministic ($0, no-LLM) seed for the axis
  retrieval benchmarks miss: across sessions, does memory keep a coding agent **safe** (Guard allows or
  blocks a proposed use by provenance + currency) and **current** (a revised decision surfaces its live
  value, not the superseded one)? Both scored end-to-end and pinned in `tests/test_continuity.py`.
- **Judged summarization A/B + rubric-coverage judge** (`eval/summarization_ab.py`) — BEAM grades
  `summarization` by a rubric (no reference string), so the prior distill A/B skipped the category
  entirely. The loader now carries the rubric (`eval/datasets.py`, `tests/test_beam_rubric.py`) and the
  new harness scores rubric coverage with a local (Ollama) or `--backend hosted` reader/judge/extractor.
  Finding: local structure-preserving extraction does not lift summarization (raw 0.28 vs replace 0.07);
  the lift is gated on a capable extractor — [`docs/frontier-2026.md §2b`](docs/frontier-2026.md).
- **TurboVec compressed vector backend** (`[turbovec]` extra) — `TurboVecStore` runs a 2/4-bit
  TurboQuant index (`TurboVecIndex` over `turbovec.IdMapIndex`) for cheap candidate generation, then
  re-scores the candidates by **exact cosine** on full-precision vectors, so recall tracks an exact
  scan while the index is ≈8–16× smaller in RAM. The RAM-saving mode (`vector_source=SQLiteVectorSource(…)`)
  keeps full vectors on disk and strips them from memory, reading back only the candidate set per query.
  Measured on BEAM-100K: recall **0.53 = exact** at both 4-bit and 2-bit; index RAM 153.6 MB → 10.2 MB
  (15×); extrapolates to 10M×768 ≈ 31 GB → ~2 GB. `examples/turbovec_backend.py`; benchmarks in
  [`BENCHMARKS.md`](BENCHMARKS.md).
- **`MemoryStore` / `VectorIndex` Protocols + allowlist pushdown** — the store/index surface is now an
  explicit contract (`midas/index.py`), and a query's scope is pushed into search as an `allowed_ids`
  allowlist (O(1) membership / in-kernel for native backends) instead of a per-row Python predicate.
- **Experimental retrieval backends (opt-in, measured).** Learned-sparse hybrid (`--midas-sparse` —
  BM42 / SPLADE++ via fastembed), ColBERT late-interaction reranker (`--midas-reranker colbert`),
  Matryoshka dim truncation (`--local-truncate-dim`), and Model2Vec static embeddings (`--model2vec`,
  `[model2vec]` extra). Per-dataset results — which generalize and which don't — in
  [`docs/MIDAS.md`](docs/MIDAS.md) §5.5 and [`docs/research-notes.md`](docs/research-notes.md).

### Changed
- **Shared, persistent memory by default.** The MCP store now defaults to `~/.midas/memory.sqlite3` (was
  ephemeral in-memory), so a fresh install just works and every client shares one memory with zero config.
  Set `MIDAS_MCP_DB=:memory:` to opt back into ephemeral. `midas/mcp_server.py`.
- **Versioned store schema with safe migration.** The SQLite store carries a schema version, auto-migrates
  on open, and refuses a store written by a *newer* Midas (with a clear `midas update` message) — so
  updates can't silently misread an evolved store. `midas/sqlite_store.py`.
- **License changed from MIT to Apache-2.0** to keep the core permissive while adding the explicit
  patent grant and enterprise-facing terms expected by commercial adopters.

## [0.0.4] — 2026-06-13

### Added
- **Distillation as one optional dial (three tiers, default no-LLM)** — the frontier's gains come
  from distilling raw turns into compact facts at ingest. Midas exposes this as a single opt-in dial,
  never a fork: **(1) default** — raw turns, $0, fully source-traceable; **(2) agent-driven** — the
  agent's own LLM distills via the injected policy + new `distill` MCP prompt ($0 to Midas, keeps raw
  + facts); **(3) local distiller** — `Memory(distiller=...)` / `OllamaDistiller` + `Memory.distill(
  turns, keep_raw=True)`, the explicitly-fenced "crosses the no-LLM line" tier for non-agentic ingest
  (keeps $0/local/zero-egress with Ollama, but distilled records are LLM-derived, stamped
  `metadata["distilled"]=True`). Default `distiller=None` = pure no-LLM. README has the tier table;
  `docs/frontier-2026.md` §2b has the research.
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
