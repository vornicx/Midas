# Overnight autonomous experiments — Midas recall / precision / speed levers

Autonomous run started **2026-06-18 ~22:05**. The user is asleep and will verify on waking.
**Nothing is committed or pushed; production defaults are unchanged; the test suite is kept green
after every code change.** Verify via: this file, `git diff` / `git status`, the per-experiment
logs under `/tmp/*.log`, and the task list.

## Context / what's already decided
- **Phase A (gate) — decided.** Widening the recall pool *hurts* Midas (fusion/recency dilution):
  `pool40/off 0.53` · `pool40/on 0.57` · `pool200/off 0.24`. Cross-encoder rerank at pool=200 is
  ~700× slower than the 42 ms baseline (killed after 3h24m). → The candidate pool is **not** a recall
  lever; the ceiling is **embedding/fusion**. TurboVec stays **capacity-only** (RAM/scale), not a
  recall booster.
- **Phase B — done, green.** `MemoryStore` / `VectorIndex` Protocols + `allowed_ids` seam + hybrid
  dedup. 191 tests pass. (`midas/index.py`, `store.py`, `ann.py`, `memory.py`, `tests/test_allowed_ids.py`.)

## Plan for this run (requested order)
1. **Embedder A/B** — bge-base vs mxbai-large vs bge-large vs jina-v3.
2. **Sparse hybrid** — SPLADE++/BM42 into the existing RRF (vs plain BM25); precision on dated/identifier queries.
3. **ColBERT rerank** — `answerai-colbert-small` late-interaction reranker at small pool.
4. **Matryoshka / Model2Vec** — dim truncation + static-embedding speed tier.

## Method (held constant unless noted)
BEAM-100K full set (`--max-convs 20` → 400 questions, 5732 events), seed 0, `--midas-only`,
deterministic recall@k (no LLM judge, reader-independent). Each embedder uses an **isolated**
embedding cache to avoid `(text_sha, dim)` collisions. **Baseline to beat: recall@k 0.53**
(bge-base, pool=40, rerank off).

---

## Results

### 1. Embedder A/B — STATUS: DONE (04:01)
| embedder | dim | recall@k | query ms | Δ vs 0.53 |
|---|---|---|---:|---:|
| BAAI/bge-base-en-v1.5 (control) | 768 | 0.53 | 49 | — |
| mixedbread-ai/mxbai-embed-large-v1 | 1024 | **0.56** | 285 | +0.03 |
| BAAI/bge-large-en-v1.5 | 1024 | 0.56 | 745 | +0.03 |
| jinaai/jina-embeddings-v3 | 1024 | **OOM (rc=137)** | — | — |

**Findings (the important part):**
- Embedder upgrade is **marginal: +0.03** (0.53 → 0.56), not the hoped ≥0.60.
- **mxbai-large dominates bge-large**: identical 0.56 recall, **2.6× faster** query (285 vs 745 ms). If upgrading at all, mxbai is the pick. But it's **6× slower than bge-base** for +0.03 — a weak recall/latency trade.
- **jina-v3 OOM-killed** (rc=137): the 2.29 GB model + corpus embedding exceeded RAM. Needs more RAM or a quantized ONNX variant; untested.
- **The ceiling is NOT the embedder — it's the `summarization` category** (recall ~0.18–0.21 across *all* models), while `knowledge_update` (~0.85) and `temporal_reasoning` (~0.86) are already strong. Bigger embedders barely move these and slightly *hurt* knowledge_update/preference. → The lever for overall recall is **multi-turn aggregation / granularity (summarization), not embedding quality.**

### 2. Sparse hybrid — DONE ⭐ THE WIN
Learned-sparse fused into the existing RRF (bge-base dense, rerank off). **BM42 = +0.10 recall** — 3× the embedder upgrade, and it gains in *every* category:

| config | recall@k | knowledge_update | preference | summarization | temporal |
|---|---:|---:|---:|---:|---:|
| dense-only (baseline) | 0.53 | 0.85 | 0.57 | 0.18 | 0.86 |
| + BM25 hybrid | 0.55 | 0.88 | 0.45 | 0.23 | 0.96 |
| **+ BM42 hybrid** | **0.63** | 0.91 | 0.68 | 0.26 | **1.00** |
| + SPLADE++ | OOM (rc=137) | — | — | — | — |

BM42 (`Qdrant/bm42-all-minilm-l6-v2-attentions`, 90 MB) beats plain BM25 cleanly — BM25 *hurt* preference (0.57→0.45); BM42 lifts it to 0.68 and takes temporal_reasoning to 1.00. ~100 s to index. SPLADE++ (0.53 GB) OOM-killed — too heavy here, so BM42 is the practical learned-sparse pick. Wired via `--midas-sparse MODEL`.

### 3. ColBERT rerank — DONE ❌ measured negative
bge-base, pool 40:

| reranker | recall@k | time | verdict |
|---|---:|---:|---|
| none | 0.53 | 8 s | baseline |
| cross-encoder (ms-marco, default) | 0.57 | 28 min | +0.04 — the only positive reranker |
| **ColBERT (answerai-colbert-small)** | **0.43** | **51 min** | −0.10, hurts every category, and slow |

ColBERT's MaxSim reorder demoted gold across the board (knowledge_update 0.85→0.66, temporal 0.86→0.71) — whether a calibration issue or a poor fit, the verdict is clear: don't ship. The cross-encoder stays the only net-positive reranker (already the default). Code kept behind `--midas-reranker colbert` for the record.

### 4. Matryoshka / Model2Vec — DONE (speed / storage tier)
| config | recall@k | note |
|---|---:|---|
| mxbai-large @1024 (full) | 0.56 | — |
| mxbai-large @512 (MRL) | 0.55 | **~lossless at half the dims** → 2× less RAM/disk, faster cosine |
| mxbai-large @256 (MRL) | 0.52 | −0.04 at quarter dims |
| model2vec potion-base-8M | 0.48 | −0.05 vs bge-base, but the whole eval ran in **5 s** (~100× faster) — a cheap candidate-gen / huge-scale tier |

mxbai is genuinely MRL-trained (512-d keeps ~98% of recall). model2vec is a viable ultra-fast tier for the candidate-pool / 10M-scale story (pairs with the TurboVec capacity plan).

---

## Status @ 05:35 — COMPLETE
All four phases ran. Headline: **BM42 sparse hybrid is the recall lever (+0.10; +0.12 with mxbai → 0.65)**;
ColBERT rerank and SPLADE were measured negatives; the structural ceiling is `summarization`. Full
numbers above, recommendation below. Code: suite 200 green, all behind flags, **no commits, defaults
unchanged**. Raw logs: `/tmp/phase234_summary.txt`, `/tmp/p234_*.log`, `/tmp/embAB_*.log`.

## Queued eval commands (sequential, after the A/B frees the CPU)
Run on the A/B winner `<W>` (point `--local-embedding-cache-path` at its A/B cache to skip
re-embedding the dense side). All BEAM-100K full set (`--max-convs 20`), seed 0, `--midas-only`.

```bash
# 2. Sparse hybrid (precision): dense-only vs BM25 hybrid vs BM42 vs SPLADE++
... --local-model <W> --midas-no-rerank --midas-hybrid                                   # BM25 hybrid
... --local-model <W> --midas-no-rerank --midas-sparse Qdrant/bm42-all-minilm-l6-v2-attentions
... --local-model <W> --midas-no-rerank --midas-sparse prithivida/Splade_PP_en_v1

# 3. ColBERT rerank (precision) at small pool=40: none vs cross-encoder vs colbert
... --local-model <W> --midas-pool 40 --midas-no-rerank                                   # baseline
... --local-model <W> --midas-pool 40                                                     # ms-marco cross-encoder
... --local-model <W> --midas-pool 40 --midas-reranker colbert                            # late interaction

# 4a. Matryoshka (speed): only meaningful if <W> is MRL-trained (jina-v3 / mxbai / nomic)
... --local-model <W> --local-truncate-dim 512
... --local-model <W> --local-truncate-dim 256

# 4b. Model2Vec (speed tier): static CPU-instant embeddings
python -m eval.runner --dataset beam --beam-tier 100K --model2vec minishlab/potion-base-8M \
  --midas-only --seed 0 --max-convs 20 --midas-no-rerank
```

## Decisions / recommendations  (read this first)

**VERDICT: BM42 was the strongest BEAM lever (+0.10) but does NOT generalize — DO NOT ship it as a default.** Validated on LongMemEval-s (n=500):

| recall@k | BEAM-100K | LongMemEval-s | query latency @246K |
|---|---:|---:|---:|
| dense-only (baseline) | 0.53 | 0.93 ✓ (≈ published 0.92) | 2.7 ms |
| + BM42 hybrid | **0.63 (+0.10)** | **0.93 (+0.00)** | **11.6 s/query** |

On BEAM (baseline 0.53, dated facts/codes) BM42's lexical signal filled real headroom and took temporal→1.00. On LongMemEval-s (baseline already 0.93, semantic/multi-session) it is a **wash** — small gains on fact/current/temporal cancel small losses on multi-session (0.91→0.89) and preference (0.91→0.88) — and the hybrid path costs **11.6 s/query at 246 K turns** (100 min total vs 75 s dense).

1. **Do NOT ship BM42 as a default.** It's a BEAM-specific win that doesn't transfer, and the hybrid path is O(N)/query at scale (the `_hybrid_candidates` `store.all()` scan + brute-force sparse scoring were never optimized — the published 0.92 was dense-only). Keep it **opt-in for BEAM-like corpora only** (low baseline recall, identifier-heavy), and only after fixing the scale cost (per-record sparse caching + an inverted-index scorer + pushing the allowlist into the hybrid path).

**The honest meta-conclusion:** no retrieval-*component* lever generalizes reliably — embedder +0.03 (and 6× slower), sparse dataset-specific (BEAM +0.10 / LongMemEval +0.00, slow at scale), rerank modest (+0.04) or negative (ColBERT −0.10). Midas's dense retrieval is already strong where the embedder captures it (0.93 on LongMemEval). The remaining gaps are **structural** (BEAM summarization / multi-turn aggregation), and the scale story is **capacity** (TurboVec, still valid for RAM) — not swapping retrieval components.

## Continuous-improvement log (autonomous loop)
- **Loop #1 — hybrid O(N)/query fix (suite 200 green).** The LongMemEval BM42 run exposed that
  `_hybrid_candidates` rebuilt `records`/`allowed`/`rec_by_id` by scanning `store.all()` *every query*
  (O(N) Python → 11.6 s/query at 246k turns). Fix: `recall()` now drops the predicate when no scope
  filter is active, and the hybrid path takes a no-filter fast path reusing a version-cached
  `_records_by_id()` map (skipping the per-query scan). Behavior-preserving; affects any `--midas-hybrid`
  use at scale (BM25 or sparse). **Validated: 46.5 ms/query (was 11,587 ms — ~250× faster)** at 246k
  turns on LongMemEval-s, recall 0.92 (BM25 hybrid). `--midas-hybrid` is now usable at scale.
- **Loop #2 — TurboVec (Phase C) de-risk.** `pip install turbovec` → 0.8.0, prebuilt wheel (12.5 MB),
  **torch-free** ✓. `IdMapIndex` API (`add_with_ids / remove / search / write / load / contains /
  prepare / dim / bit_width`) matches the Phase-B `VectorIndex` Protocol → a `TurboVecIndex` adapter is
  feasible. Next: verify method signatures (allowlist, uint64 ids) and scaffold the adapter +
  exact-rerank-from-SQLite (compressed candidate-gen in RAM, full vectors from the SQLite BLOBs).
- **Loop #3 — `TurboVecIndex` adapter built (suite 204 green).** `midas/turbovec_index.py` wraps
  TurboVec's `IdMapIndex` behind the `VectorIndex` Protocol, owning the stable `str ↔ uint64` id map;
  add_with_ids / search(allowlist) / remove / write+load all tested against the real wheel
  (`tests/test_turbovec_index.py`). API confirmed: `search(queries_2d_f32, k, allowlist=uint64[]) →
  (scores, ids)`; `prepare()` before search. Next: `TurboVecStore` (MemoryStore) using it +
  exact-rerank from full vectors, then a recall/RAM benchmark vs exact/IVF (Phase D).
- **Loop #4 — `TurboVecStore` built (suite 208 green).** `midas/turbovec_store.py` implements the
  `MemoryStore` contract via `TurboVecIndex` for compressed candidate-gen + an exact-cosine rerank on
  full vectors (the recall safety net). Scope (allowlist/predicate) is pushed into the search; tested
  (`tests/test_turbovec_store.py`): exact rerank recovers the self-match as #1, allowlist/predicate
  filtering, MemoryStore conformance. `Memory(store=TurboVecStore(dim=…))` now works. Next (Phase D):
  wire `--midas-store turbovec` and benchmark recall + RAM vs exact/IVF on BEAM-100K.
- **Loop #5 — Phase D: `--midas-store turbovec` wired + recall benchmark running.** Adapter/runner
  now build `Memory(store=TurboVecStore(dim, bit_width, rerank_pool))` via `--midas-store turbovec
  --midas-turbovec-bits {2,4}`. Smoke OK (matches exact recall on the 5q sample, ~70 ms/query).
  **Phase C/D VALIDATED — TurboVec preserves recall AND compresses RAM.** Benchmark (BEAM-100K full
  set, bge-base, store does the exact rerank):

  | store | recall@k | query ms | index RAM (50k×768) |
  |---|---:|---:|---:|
  | exact | 0.53 | 12 | 153.6 MB (float32) |
  | TurboVec 4-bit | 0.53 | 50 | 19.8 MB (**7.8×**) |
  | TurboVec 2-bit | 0.53 | 48 | 10.2 MB (**15×**) |

  **Recall identical (0.53) even at 2-bit / 16× compression** — the exact-cosine rerank fully recovers
  the order, so quantization only generates candidates. Extrapolated to **10M × 768: 30.7 GB float32 →
  1.9 GB at 2-bit** → long-horizon memory on a laptop. TurboVec is slower than exact at the 100k tier
  (exact scan is trivial there); its win is RAM/capacity at 1M–10M, as the thesis predicted.
  **Caveat:** the first-cut `TurboVecStore` still keeps full embeddings in RAM for the rerank, so the
  *end-to-end* RAM win needs the next increment — a SQLite-backed store that drops the in-memory
  float32 and reads full vectors from the BLOB column only for the candidate set.
- **Loop #6 — `SQLiteVectorSource` built (suite 213 green).** `midas/vector_source.py`: an on-disk
  `id → embedding` store (SQLite float32 BLOB; batched `get_many` that chunks past the 999-variable
  limit) — the full-precision side of compressed search, so the rerank reads only the candidate
  vectors from disk instead of holding the float32 matrix in RAM. Tested (roundtrip, upsert, remove,
  persistence, 2.5k batch). Next: wire `vector_source` into `TurboVecStore` (strip in-RAM embeddings,
  rerank-fetch from disk) → the end-to-end RAM win.
- **Loop #7 — RAM-saving `TurboVecStore` complete (suite 214 green).** `vector_source` mode: the index
  is built **incrementally** and full embeddings are stripped from the in-RAM records once indexed
  (persisted to `SQLiteVectorSource`); the exact rerank reads only the candidate vectors back from
  disk, and `get()` re-attaches on demand. Tested: search-from-disk still recovers self as #1,
  `_records[id].embedding is None` after indexing, `get()` re-attaches, delete cascades to disk. So
  steady-state RAM = the compressed index alone (**no float32 matrix**) — the long-horizon-on-a-laptop
  goal realized end-to-end with recall intact. *Limitation:* embedding-dependent extras (anchor_boost
  / novelty / supersede) would need the same on-demand fetch; the recall→context path does not.
  **TurboVec arc complete: recall-validated, RAM-compressed (15×), end-to-end RAM-saving, tested.**
- **Loop #8 — TurboVec packaged + usage example (shippable).** Added `[turbovec]` and `[model2vec]`
  extras to `pyproject.toml` (and `all`), plus `examples/turbovec_backend.py` — validated end-to-end
  with the real bge-base embedder: `Memory(store=TurboVecStore(bit_width=2, vector_source=…))` recalls
  the gold fact at 0.986 with embeddings stripped to disk. TurboVec is now complete **and shippable**:
  recall-validated, 15× RAM-compressed, end-to-end RAM-saving, tested (suite 214), packaged, documented.
- **Loop #9 — RAM-saving mode correctness (suite 215 green).** `TurboVecStore.search` now re-attaches
  each returned hit's full vector (only the top-`limit`, reusing the rerank fetch), so callers reading
  `hit.record.embedding` (anchor_boost, novelty) behave the same as with an exact store while the
  stored records stay stripped — resolves the Loop #7 limitation for the recall path.
- **Loop #10 — CHANGELOG consolidated.** `[Unreleased]` now records the shippable loop output: the
  hybrid O(N) fix (Fixed), the TurboVec backend + Protocols/allowlist (Added), and the experimental
  retrieval backends (Added, opt-in). The repo is review-ready: `git diff` + `CHANGELOG.md` +
  `docs/overnight-experiments.md` tell the whole story; suite 215 green; no commits; defaults unchanged.
2. **Embedder: mxbai-large is the best upgrade** (= bge-large recall at 2.6× the speed; jina-v3 OOMs), but only +0.03 alone / +0.02 on top of BM42, at **6× the query latency**. A recall/latency judgment call — keep bge-base as the fast default, offer mxbai for max recall.
3. **Negatives, kept (eval-first):** ColBERT rerank −0.10 (and 51 min); SPLADE++ OOM; pool widening −0.29 (Phase A); a bigger embedder alone is marginal.
4. **Speed/storage tier:** mxbai@512 ≈ lossless (2× smaller); model2vec 0.48 at ~100× speed — a cheap pre-pool / 10M-scale option that pairs with TurboVec (Phase C, still valid for RAM/capacity, not recall).
5. **The structural ceiling is `summarization`** (multi-turn aggregation): 0.18 → 0.28 with mxbai+BM42 — materially better but still the frontier. The next real recall gain is aggregation/granularity, not retrieval components.

**Next steps (need you awake):**
- **Validate BM42 on LongMemEval-s + LoCoMo** before shipping — does +0.10 generalize beyond BEAM? (Likely, given the per-category breadth, but measure.)
- **Stack BM42 + cross-encoder:** `hybrid` and `rerank` are currently *mutually exclusive* in `recall()` (hybrid uses RRF, skips the cross-encoder). A small change to rerank the hybrid pool could add the cross-encoder's +0.04 on top of 0.65 → possibly ~0.67+. Worth a flag.
- **Production sparse:** the BM42 index rebuilds (re-embeds the corpus) on every store change — fine for read-heavy eval; production needs per-record sparse-vector caching + incremental update.
