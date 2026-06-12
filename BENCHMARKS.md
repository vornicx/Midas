# Midas Benchmarks

Honest, reproducible benchmarks for the Midas agentic-memory SDK. Every number here comes from a
real run with the command to reproduce it. We deliberately **lead with reader-independent metrics**
(retrieval + cost) and treat end-to-end answer correctness as a secondary, noisy signal — see
[`docs/methodology.md`](docs/methodology.md) for why that is the honest choice, not a convenient one,
plus the verbatim MCP policy, failure-case traces, and how conflicting memories are handled.

## TL;DR

Midas isolates and wins the two axes that actually measure a *memory layer* (as opposed to the
reader LLM stacked on top):

- **Retrieval** — on the **full** LongMemEval-`s` set (evidence buried among distractors, all 500
  questions), Midas retrieves the supporting turns at **recall@k 0.92** vs a recency-window
  baseline's **0.01**.
- **Cost** — Midas does **0 LLM calls, $0 API spend, and 0 data egress at ingest** (local embeddings
  only), versus LLM-at-ingest memory systems that call an LLM per session to extract facts.

## 1. Retrieval quality — `recall@k` + `precision@k` (deterministic)

| dataset | loader | what it stress-tests |
|---|---|---|
| `synthetic-v0` | `eval.datasets.synthetic` | needles in chatter; offline, zero setup |
| `locomo10` | `eval.datasets.locomo` | real long conversations (LoCoMo) |
| `longmemeval-{s,m,oracle}` | `eval.datasets.longmemeval` | buried evidence, multi-session, temporal (ICLR 2025); use [cleaned HF release](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) → `data/longmemeval_s.json` |
| `multiday-v1-adversarial` | `eval.datasets.multiday` | evolving facts, stale answers, unanswerable Qs |
| `conflicts-v1` | `eval.datasets.conflicts` | near-duplicates, temporal conflicts, entity-confusable pairs |

All loaders share one schema (`eval/schema.py`); the same runner, multiday, and retention harnesses apply.

Fraction of gold supporting turns retrieved into the context. Fully deterministic (local embeddings,
no LLM), so it reproduces exactly.

| dataset | setting | baseline-raw | **Midas** |
|---|---|---:|---:|
| **LongMemEval-`s`** (FULL set: 500 questions, 246,750 turns ingested) | n=500, bge-base, no rerank, seed 0 | 0.01 | **0.92** |
| **LoCoMo** (FULL public set: 10 conversations, 5,882 turns) | n=1,540, bge-base, no rerank, seed 0 | 0.05 | **0.73** |

Across **both** full datasets Midas retrieves the supporting turns at **0.73–0.92** while a recency
window gets **≤0.05** — the wedge holds beyond a single benchmark, with **no sampling caveat left**:
these are the complete public question sets.

> **Correction (2026-06-10).** An earlier version of this table reported LoCoMo recall@k **0.85**
> on an n=50 sample. That number does **not reproduce** against the publicly downloadable
> `locomo10.json` with the documented config — we verified it isn't code drift by re-running the
> v0.0.1 harness on today's dataset (same result). Re-measuring end to end also exposed that the
> runner's old LoCoMo-specific `min_relevance=0.75` floor — tuned on that early sample — prunes
> most gold turns on the full set (recall@k **0.18** with it, **0.73** without), so it has been
> removed; the SDK's scale-free ratio floor replaces it. The number above is the full public set,
> nothing held out, reproducible with the command below.

On the full LongMemEval-`s` set the per-category recall@k is strong across the board: fact **0.97**
(n=126) · knowledge-update **0.93** (n=78) · temporal **0.91** (n=133) · multi-session **0.89**
(n=133) · preference **0.89** (n=30). A recency window finds essentially **none** of the buried
evidence; Midas finds ~9 in 10 — exactly the multi-session setting where retrieval quality decides
whether the answer is even *possible*. (Full-run cost, measured on the same CPU box: 246,750 turns
ingested at ~61 ms/event, ~30 ms/query, zero LLM calls and $0.)

The harness also reports **`precision@k`**: the fraction of retrieved source turns that are actually
gold supporting evidence. This is the false-positive stress metric for agent loops: a memory layer can
hit the buried fact and still hurt the agent if it packs tempting distractors beside it. Use
`--trace-questions --trace-snippets` to publish concrete gold-vs-retrieved examples, including failure
cases.

### Dumb-reader ablation — proving the numbers aren't reader-inflated

A standing critique of memory benchmarks is that a capable reader does "heroic recovery", making weak
retrieval look strong. The `--dumb-reader` flag adds a **deterministic extractive reader** (no LLM):
it picks the single retrieved turn with the most content-word overlap with the question and scores it
against gold (verbatim hit = 1.0, else token F1). It cannot reason, combine turns, or paraphrase — so
`answer_dumb` can only move with retrieval quality:

| dataset (offline, hashing embedder) | adapter | recall@k | answer_dumb |
|---|---|---:|---:|
| synthetic-v0 | baseline-raw | 0.50 | 0.38 |
| synthetic-v0 | **Midas** | **1.00** | **1.00** |
| conflicts-v1 | baseline-raw | 0.78 | 0.49 |
| conflicts-v1 | **Midas** | **1.00** | 0.82 |

`answer_dumb` tracks `recall@k` for both systems — the gap between systems is the memory layer, not
the reader. See [docs/methodology.md](docs/methodology.md) for the per-question failure cases this
ablation exposes (the dumb reader quoting a stale repetition is the best argument for supersession).

```bash
python -m eval.runner --dataset synthetic --dumb-reader
python -m eval.runner --dataset conflicts --dumb-reader --midas-supersede
```

**Time-aware retrieval (LLM-free).** Memories carry real **event time** (parsed from the dataset's
session timestamps), so recency and chronological context reflect *when things happened*, not load
order — the bitemporal signal long-horizon memory needs. Turning it on lifts **temporal recall@k
0.86 → 0.95** (n=40, A/B via `--midas-no-time`), in line with the LongMemEval paper's +7–11% for
temporal handling — but done with regex/relative-date math, **not** an LLM, preserving the no-LLM
ingest/query edge. (fact dips 0.92 → 0.89, within n=13 noise and with no effect on fact *answer*.)

```bash
# reproduce (deterministic; downloads LongMemEval-s on first run)
# Dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
# Place longmemeval_s_cleaned.json at data/longmemeval_s.json (or use --longmemeval-path)
python -m eval.runner --dataset longmemeval --variant s --local \
  --local-max-text-chars 600 --local-batch-size 16 --midas-no-rerank \
  --max-questions 500 --limit 20 --seed 0      # full set; ~4 h cold on a modest CPU box

# LoCoMo, FULL public set (download data/locomo10.json from github.com/snap-research/locomo)
python -m eval.runner --dataset locomo --max-convs 10 --local --midas-no-rerank --seed 0
```

### Context parsimony — a scale-free relevance floor (default on)

A memory layer can hit the buried fact and still hurt the agent by packing distractors beside it —
our `precision@k` hovered at 0.06–0.16 because recall-tuned top-k retrieval drags in topically-near
noise. The fix that survived measurement: drop any hit whose relevance is **< 0.3× the query's own
top hit** (`min_relevance_ratio`, default 0.3, `0` disables). Being *relative*, it transfers across
embedders — unlike an absolute floor, whose useful value depends on each embedder's cosine scale.
Measured (deterministic, dumb-reader):

| dataset (hashing) | recall@k | precision@k | avg_tokens |
|---|---:|---:|---:|
| synthetic — off → **0.3** | 1.00 → **1.00** | 0.15 → **0.20** | 102 → **87** |
| conflicts — off → **0.3** | 1.00 → **1.00** | 0.08 → **0.18** | 207 → **121** |
| multiday — off → **0.3** | 1.00 → **1.00** | 0.09 → **0.16** | 174 → **126** |

Zero gold evicted anywhere, ~2× precision, ~30–40% fewer context tokens. On LongMemEval-`s`
(bge-base, n=40) it is a measured **no-op** (recall@k 0.95 and tokens identical) — bge's compressed
cosine scale keeps everything above 0.3× the top hit, which is exactly the safety property a
*relative* floor buys. The honest boundary: **0.4+ is too aggressive** — it evicts multiday's buried
low-similarity update (recall@k 1.00 → 0.80), the same failure mode that keeps hybrid opt-in. So the
default stays at the measured-safe 0.3.

```bash
python -m eval.runner --dataset multiday --dumb-reader --midas-supersede \
  --midas-min-relevance-ratio 0   # the A/B: disable the floor
```

### BEAM — the 10M-token frontier benchmark (first results)

[BEAM](https://arxiv.org/abs/2510.27246) (ICLR 2026) is the field's successor to LoCoMo/LongMemEval:
coherent conversations at **128K–10M tokens** where context-stuffing is physically impossible, with
ten memory abilities and `source_chat_ids` evidence — so the deterministic `recall@k` applies (see
[`docs/frontier-2026.md`](docs/frontier-2026.md) for the landscape). First tier, full set
(20 conversations, 5,732 turns, 400 questions, bge-base, no rerank, seed 0):

| BEAM 100K (deterministic) | baseline-raw | **Midas** |
|---|---:|---:|
| recall@k (overall) | 0.00 | **0.56** |
| ingest cost | — | ~91 ms/turn local, **0 LLM, $0** |

Per-category recall@k tells an honest structural story. Midas is strongest exactly where its
belief machinery lives — **temporal 0.90 · knowledge-update 0.89 · contradiction-resolution
0.80** — mid on retrieval breadth (preference 0.65 · extraction 0.60 · multi-session 0.51), and
weak on the aggregation abilities (**instruction-following 0.26 · event-ordering 0.24 ·
summarization 0.18**), which need *the whole conversation*, not the top-k turns — the documented
cost of the no-LLM/no-summarization trade. A recency window retrieves **0.00 across every
category**: at 100K tokens there is no naive fallback. The 500K/1M/10M tiers are next (ingest is
embed-bound, so the 10M tier costs hours of local CPU — not the dollars-per-conversation an
LLM-at-ingest pipeline pays there).

**Pinned standing directives (measured fix for the instruction gap).** A durable user rule
("from now on, reply in Spanish") applies to *every* turn yet is semantically unrelated to most
queries, so relevance-ranked recall structurally misses it. Midas now detects standing directives
at ingest (cue regex, no LLM, user-voiced turns only) and `build_context` pins up to
`MIDAS_MCP_PINNED` of them into every context — the no-LLM version of Letta's always-in-context
core memory. Measured on BEAM-100K (full tier): **instruction-following recall@k 0.26 → 0.44**
(pinned=2, the default) **→ 0.51** (pinned=4), overall 0.56 → **0.58**, no real regression
elsewhere (extraction −0.03, within noise). The first iteration of the detector — which also
pinned assistant-voiced advice — *hurt* across the board (overall 0.50, instructions 0.21) and is
kept as the honest negative that set the user-voice-only rule.

```bash
python -m eval.runner --dataset beam --beam-tier 100K --max-convs 20 --local \
  --midas-no-rerank --dumb-reader --seed 0 --midas-only --midas-pinned 4   # the A/B
```

```bash
# data/beam/100K.parquet from https://huggingface.co/datasets/Mohammadta/BEAM
python -m eval.runner --dataset beam --beam-tier 100K --max-convs 20 --local \
  --local-max-text-chars 600 --midas-no-rerank --dumb-reader --seed 0
```

### Multilingual embeddings — measured both ways (opt-in)

The default `bge-base` is **English-only**, and that failure mode is silent: on a Spanish mirror of
the synthetic dataset (`--dataset synthetic-es`) its deterministic extractive answer drops to
**0.68**, while `paraphrase-multilingual-MiniLM-L12-v2` (`MIDAS_MCP_EMBEDDER=multilingual`, ~120 MB,
384-d) scores **1.00** — and both score 1.00 on the English twin. The other side of the trade,
measured on English LongMemEval-`s` (n=40, deterministic): the multilingual model retrieves at
**recall@k 0.83 vs bge-base's 0.95**. So: work in English → keep the default; work in another
language → switch to `multilingual`, where the default effectively fails. (The cross-encoder
reranker is English-trained and stays off in multilingual mode.)

```bash
python -m eval.runner --dataset synthetic-es --local --dumb-reader --midas-no-rerank \
  --local-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2   # vs default bge
```

### Hybrid retrieval (BM25+RRF) — measured, kept opt-in

Fusing a lexical BM25 ranking with semantic recall (`recall(hybrid=True)`, reciprocal-rank fusion)
is the standard "push past the embedding ceiling" lever — so we measured it instead of defaulting
it. On **conversational long-horizon data it is a negative**: LongMemEval-`s` (n=40, bge-base,
deterministic A/B) drops multi-session recall@k **0.97 → 0.81** and temporal **0.95 → 0.86** (fact
0.89 → 0.90, noise) — lexical rank-fusion displaces buried semantic evidence on paraphrased
queries. On synthetic and conflicts-v1 it ties at recall@k 1.00. So hybrid stays **off by
default**; turn it on per-query when the query is an exact identifier (an error code, a ticket id,
a function name) rather than a paraphrase — the case where BM25 catches what the bi-encoder ranks
low. The BM25 index is cached on the store's change counter with per-term posting lists, so on a
stable 5k-record store a hybrid query costs **~8 ms** (vs ~66 ms when it was rebuilt per query);
eval metrics are bit-identical to the uncached version.

```bash
python -m eval.runner --dataset longmemeval --variant s --local --midas-no-rerank \
  --max-questions 40 --limit 20 --seed 0 --midas-only --midas-hybrid   # the A/B
```

## 2. Cost / latency — the no-LLM edge (memory layer only)

Measured with the runner's cost instrumentation; excludes the shared reader/judge LLM (identical
across systems).

| system | ingest ms/event | memory-layer LLM | API $ | data egress |
|---|---:|---|---|---|
| **Midas** | ~116 (cold) · ~0 (cached) | **0** | **$0** | **none** |
| Mem0 *(LLM-at-ingest class)* | ~668 | ≥1 call / session | yes (per token) | yes (every turn) |

Midas's ingest cost is pure local ONNX embedding. LLM-at-ingest systems (Mem0, and **Hindsight**,
whose TEMPR extracts facts with an LLM at `retain` and CARA reasons with an LLM at `reflect`) pay an
LLM call per ingested session — which means **$/token forever at scale, seconds of latency, and every
conversation turn leaving the box**. At the scale where agent memory actually matters, that cost
structure — not a few points of benchmark accuracy — is what decides build-vs-buy.

**Microbenchmark (`eval/bench_perf.py`, bge-base on a modest CPU box — measured, not estimated):** a
single `remember` is **~16 ms p50** on short records (embed-bound — there is no LLM, just the ONNX
embedding forward pass; the time scales with text length, so the longer real turns above land at the
~116 ms/event figure), `build_context` **~51 ms p50** over a 2,000-record store; ingest batches far
faster via `remember_many`. Honest framing: these are **tens of milliseconds, embed-bound** — fast and local (no
per-turn network round-trip), not sub-millisecond. **Footprint: ~3.6 MB per 1,000 records** at 768 dims
— embeddings are stored as **float32 arrays**; a Python `list[float]` would cost ~7× more (~24 MB/1k),
and switching to float32 left `recall@k` unchanged (LoCoMo 0.27 → 0.27). SQLite persistence already
stored float32.

```bash
python -m eval.bench_perf --local --n 2000 --q 200   # latency · throughput · real (tracemalloc) footprint
```

**Every Midas mechanism is local, $0, zero-egress** — embeddings (bge-base ONNX), recall, supersession,
the NLI contradiction/entailment checks (`midas/nli.py`, int8 ONNX MNLI), and the abstention metric.
The only LLM is the *reader*, which is pluggable. **Demonstrated end-to-end fully offline** — Midas +
a local `llama3.2:1b` reader/judge via Ollama (on a local GPU): LongMemEval-`s` n=10 → recall@k **0.80**,
answer **0.40**, **0 API calls / $0 / nothing leaves the box**. (The modest answer rate is the 1B
reader; a larger local model lifts it — correctness is reader-bound. The point: the pipeline is 100%
local at zero marginal cost.)

```bash
# fully-local, $0, offline pipeline (local Ollama reader; no API key):
ollama serve &   # then: ollama pull llama3.2:1b
python -m eval.runner --dataset longmemeval --variant s --local --midas-no-rerank \
  --judge --judge-ollama --judge-model llama3.2:1b --max-questions 10 --seed 0
```

```bash
# reproduce (cold ingest cost: disable the embedding cache)
python -m eval.runner --dataset longmemeval --variant s --local \
  --no-local-embedding-cache --midas-no-rerank --max-questions 3 --limit 20 --seed 0
```

## 3. Provenance (auditability)

`recall@k` is computable for Midas and the recency baseline because they return **source turn IDs**.
It is **N/A for fact-synthesizing systems** (Mem0, Hindsight) — they return LLM-rewritten facts, not
traceable sources. For Midas this is a feature: retrieved context is **auditable back to the exact
source turn**, with no extraction-time LLM that can silently hallucinate. This matters for enterprise
and compliance.

## 4. Scaling — sub-linear search past the exact scan (ANN)

The default `InMemoryStore` runs an **exact** cached cosine scan: O(N) per query but fast in absolute
terms (~5 ms/query at 36k × 768-d; ~130–230 ms extrapolated to 1M). For larger corpora, `IVFStore`
wraps a **numpy-only** inverted-file index — **no native dependency** (unlike faiss/hnswlib): the
corpus is k-means-clustered into `nlist` cells and a query scans only the `nprobe` nearest cells, so
search is **sub-linear**. `nprobe` tunes recall vs latency at query time, with no rebuild.

Measured on the **36k real bge-base embeddings** cached from the runs above (k=10, 500 held-out
queries — real embeddings cluster, which is IVF's intended regime; uniform-random vectors are its
worst case):

| nprobe | recall@10 vs exact | IVF ms/q | speedup vs exact |
|---:|---:|---:|---:|
| 1  | 0.52 | 0.13 | 37× |
| 4  | 0.82 | 0.76 |  6× |
| 8  | 0.91 | 1.49 |  3× |
| 16 | 0.95 | 3.27 |  1.5× |

The win **grows with N** (IVF scans ≈ `nprobe·√N` candidates vs exact's N): the exact↔IVF crossover
is ~10k records, and at nprobe=8 the speedup rises 0×→1×→2×→3× across 5k→36k. Extrapolating by the
candidate count, at **1M** records exact ≈ 130–230 ms vs IVF(nprobe=8) ≈ 8 ms — **~20× at recall
~0.90**. Below ~10k the exact scan wins (clustering overhead dominates), which is why `InMemoryStore`
stays the default and `IVFStore` is opt-in for large, read-heavy corpora.

```bash
python -m eval.bench_ann   # real cached embeddings if present, else synthetic clustered
```

## 5. Retention — selective forgetting beats recency (no LLM)

Long-horizon memory must stay **bounded**. Midas forgets by `memory_value` (importance × recency); the
real question is whether that keeps the *right* memories under pressure. Measured on **LongMemEval-`s`
(n=40, evidence buried among distractors)** by evicting to a fixed budget and comparing policies
(`eval/retention.py`, recall@k averaged over all 40 questions):

| keep | importance — `StructuralImportance` | importance — `ContentImportance` | recency (FIFO) | random |
|---|---:|---:|---:|---:|
| 50% | **0.56** | 0.43 | 0.36 | 0.25 |
| 25% | **0.36** | 0.26 | 0.19 | 0.12 |

**Importance-ranked forgetting beats recency at every level** (`value > fifo > random`), and a structural
salience signal (boost an *assertion of a durable attribute*; demote questions/meta) beats the plain
content score by **+0.10–0.13 recall@k** under forgetting — all **no-LLM**. On undifferentiated chat
(LoCoMo) the signal is neutral, because recall there doesn't gate on importance; the buried-fact setting
is where it shows. (The honest negatives along the way — *novelty-vs-store* and *reinforcement* — are
documented in the design doc; the moat is reporting them too.)

```bash
python -m eval.retention --dataset longmemeval --variant s --local --no-rerank \
  --structural-importance --value-rank-only --max-questions 40 --fractions 0.5,0.25
```

**Per-question forgetting traces (`--trace`).** Averages hide which facts died. The `--trace` flag
prints a per-question audit after eviction — which gold turns survived, which were evicted, and the
value-vs-fifo diff — yielding publishable success *and failure* examples. From a real run (multiday,
offline): at keep 25–75% fifo evicts old-but-durable facts that value keeps (`'The primary database
is PostgreSQL.'`); and in the pure rank-only configuration (protections off) value itself evicts the
**buried low-importance updates** (`"we'll actually go live the first week of October instead"`) —
the measured reason the shipping config protects the durable tier. Full traces in
[docs/methodology.md](docs/methodology.md).

```bash
python -m eval.retention --dataset multiday --trace                     # shipping behaviour
python -m eval.retention --dataset multiday --trace --value-rank-only   # exposes the failure mode
```

## 5b. Adversarial near-duplicates & temporal conflicts — `conflicts-v1`

Local embedding memory often looks clean until the corpus contains *"same fact, different date"* or a
repeated plan with one constraint changed. `conflicts-v1` encodes those traps deliberately: the stale
value is repeated 2–3× with near-identical prominent phrasing while the update appears once, buried;
a multi-clause deploy plan is restated verbatim except one number; and entity-confusable pairs
(Apollo/PostgreSQL vs Artemis/MySQL) where **both facts are true and both are asked** — wrongly
superseding one with the other fails a question. Deterministic (no LLM, `--context-only`):

| adapter | ctx_current | ctx_stale | ctx_contradict | avg_tokens |
|---|---:|---:|---:|---:|
| Midas (supersede=on) | 1.00 | **0.86** | **0.86** | **90** |
| Midas (supersede=off) | 1.00 | 1.00 | 1.00 | 112 |

With belief revision off, every updated fact drags its stale twin into context (the adversarial
construction works). Supersession retires stale copies (0.86) and shrinks context **without losing a
single current value and with zero wrongful entity supersessions** (all four confusable questions
keep recall@k 1.00) — because revision is gated on entity overlap + ambiguity margin, not cosine
alone. The residual 0.86 is the honest gap. `ctx_stale`/`ctx_contradict` also surface in the main
runner leaderboard for any dataset annotating outdated values.

```bash
python -m eval.multiday --dataset conflicts --context-only --ab-supersede --midas-only
python -m eval.runner --dataset conflicts --dumb-reader --midas-supersede
```

## 6. Correctness with a fixed strong reader (secondary)

`recall@k` measures the memory layer directly; *answer correctness* additionally depends on the reader
LLM (see Methodology). Holding the reader **fixed and identical across systems** (`gpt-4.1-mini` at
temp 0 — the same non-reasoning reader class the LongMemEval leaderboard uses), Midas's retrieval edge
converts to a large answer edge:

| dataset (reader = gpt-4.1-mini) | baseline-raw answer | **Midas** answer |
|---|---:|---:|
| LongMemEval-`s` (n=40, seed 0) | 0.05 | **0.82** |

Per-category Midas answer (indicative, wide bars at n=4–13): fact **1.00** · knowledge-update **1.00** ·
multi-session **0.89** · temporal **~0.64–0.82** (noisy) · preference 0.33. Same reader for both, so the
~16× gap (0.82 vs 0.05) is the memory layer: a recency window almost never holds the buried evidence
(recall@k 0.03), so the reader cannot answer.
For scale, 2026 SOTA on LongMemEval is **reader-dominated and LLM-ingest-based**: Mastra Observational
Memory scores **84.2% (gpt-4o) → 94.9% (gpt-5-mini)** — a +11pt swing from the *reader alone* — using
an LLM Observer/Reflector at ingest; Mem0 ~94.4.

**Reader sweep — same reader as SOTA, but Midas does ZERO LLM at ingest** (LongMemEval-`s`, n=40,
seed 0; judge fixed = gpt-4o to match Observational Memory's protocol; structured answerer):

| reader (Midas, no-LLM ingest) | **Midas** answer | Observational Memory (LLM ingest) |
|---|---:|---:|
| gpt-4o | **0.84** | 0.84 — **match** |
| gpt-5-mini | 0.87–0.89 | 0.95 |

**At gpt-4o, Midas ties the SOTA (0.84) with $0 LLM ingest** — OM gets the same number by running an
LLM Observer+Reflector on every conversation at ingest. With gpt-5-mini Midas reaches 0.87–0.89 vs
OM's 0.95: OM's curated observations help a strong reader more than raw retrieved turns do. Across the
sweep Midas pays **$0 at ingest, no data egress, and returns source-traceable turns** — none of which
the LLM-ingest systems offer — and its retrieval **scales to ~500-session corpora (LongMemEval-`m`) by
retrieving + forgetting** (measured: a ~4,944-turn haystack assembles a bounded ~480-token context,
recall@k 0.78 over n=3), where a keep-every-observation-in-context design does not fit by construction.
*(We measure Midas's side at that scale; we do not run OM — the overflow is an architectural inference,
not a head-to-head.)* By category Midas **leads
multi-session (0.89 vs OM's 0.872)** and matches knowledge-update (1.00 vs 0.962); the remaining gap is
**temporal** (0.82 vs 0.955; per-category n=4–13 → wide bars).

A **structured answerer** (ask the reader to pull the relevant dated entries and do the date arithmetic
before answering) lifts **non-reasoning** readers a lot — gpt-4o 0.76 → 0.84, multi-session 0.56 → 0.89
— and is neutral for reasoning readers that already do this internally (gpt-5-mini 0.89 → 0.87, within
noise). Since the cheap, deployable readers are the non-reasoning ones, it is on by default.

**Time-awareness — measured on the deterministic metric.** The LLM-free event-time grounding lifts
**temporal `recall@k` 0.86 → 0.95** (A/B via `--midas-no-time`, deterministic and reproducible) with
no real regression elsewhere (multi-session holds at 0.97; fact 0.92 → 0.89 is within n=13 noise). Its
effect on *answer* correctness is real in principle — the reader can resolve "how many days ago…" from
the dated context + a "today" anchor — but at **n=11 per category the answer deltas are inside run-to-run
judge noise** (the temporal answer alone bounced 0.64–0.82 across identical-config runs), so we do
**not** quote a per-category answer lift. This is the methodology working as intended: trust `recall@k`,
distrust small-n correctness deltas.

**Caveat:** n=40 sample with gpt-4.1-mini; the published Zep/Mem0 numbers run their full systems over
the full set with GPT-4o. Correctness also moves far more with the reader than with the memory layer
(see Methodology) — a strong reader can still miss multi-hop reasoning even when recall@k is high (the
evidence is present; the reasoning is the bottleneck). So we treat correctness as a secondary, wide-bar
signal and lead with `recall@k`.

```bash
# reproduce (needs an LLM key; this used OpenRouter gpt-4.1-mini as reader + judge)
JUDGE_PROVIDER=openrouter JUDGE_MODEL=openai/gpt-4.1-mini \
python -m eval.runner --dataset longmemeval --variant s --local \
  --local-max-text-chars 600 --local-batch-size 16 --midas-no-rerank \
  --judge --max-questions 40 --limit 20 --seed 0      # add --midas-no-time for the A/B
```

## Methodology — why reader-independent metrics

End-to-end "answer correctness" on these benchmarks is **dominated by the reader LLM, not the memory
layer** (full write-up, anti-cheating checklist, dumb-reader ablation, conflicts-v1 stress tests,
forgetting traces, supersession mechanics, and verbatim MCP policy text:
[`docs/methodology.md`](docs/methodology.md)):

- Holding the reader fixed, a memory layer's lift is real; but swapping in a bigger reader moves the
  *headline* far more than the memory does. (Public SOTA on LongMemEval reports ~39% → ~83% from the
  memory system but ~83% → ~91% from *just a larger reader* — most of the headline is the reader.)
- Our own hosted LLM judge (an MoE served via API) is **not reproducible across sessions**: identical
  inputs scored ~0.46 one day and ~0.13 the next, even at temperature 0. We added a local,
  seed-pinned, serialized judge (`--judge-ollama`) to make correctness reproducible, but a small local
  reader is too weak to *use* good context — so correctness still does not cleanly isolate memory
  quality.

Therefore: **`recall@k`, `precision@k` (deterministic, reader-independent) and ingest cost
(structural) are our primary metrics.** We report correctness only with a fixed reader and wide error
bars, and never as a headline.

### Honest caveats
- **Retrieval headlines carry no sampling caveat**: LongMemEval-`s` is the full 500-question set
  and LoCoMo is the full public set (10 conversations, n=1,540). Judged answer-correctness
  (sections below) remains n=40 — it needs a hosted reader/judge per question, which is
  cost-bound, not methodology-bound.
- **Latency is hardware/provider-dependent** (the ~668 ms for the LLM-at-ingest class includes API
  round-trip). The durable, hardware-independent claim is the **0-LLM / $0 / no-egress** column.
- **baseline-raw** = "stuff recent turns into the window" (the naive big-context approach).
- Numbers measured on CPU with `BAAI/bge-base-en-v1.5`. GPU / a faster embedder lowers Midas latency.
- **Reranking is off by default on large haystacks.** A cross-encoder reranker is available, but on
  LongMemEval-`s` (CPU) it added ~80× query latency (4.2 s vs 53 ms) with **no `recall@k` change**
  (0.88 → 0.88): it reorders the records that already fit the budget (which can help the *reader*) but
  does not change *which* evidence fits. So it is not on the retrieval-quality path here.

### Failure-case traces and policy text

For useful external review, publish the trace table, not just aggregate scores:

```bash
python -m eval.runner --dataset longmemeval --variant s --local --midas-no-rerank \
  --trace-questions --trace-snippets --max-questions 40
```

The trace shows `gold`, `retrieved`, `recall@k`, `precision@k`, and snippets for each question, so
selective-forgetting wins and failures are inspectable instead of hidden behind an average. The exact
MCP-injected memory policy is also exposed at runtime via the `memory_policy` tool; it includes the
provenance labels and the Guard rule that external/destructive actions require `user_confirmation`.

**[docs/methodology.md](docs/methodology.md)** collects all of this in one place: what the pipeline
does *not* do (no query rewriting, no LLM at ingest, no gold leakage, seeded sampling), the
dumb-reader ablation, the conflicts-v1 results, how conflicting memories are handled (supersession
chains + gating), real failure cases with traces, and the **verbatim MCP-injected policy text**.

*All commands run from the repo root. `recall@k` / `precision@k` require no API key; `--judge*` flags do.*
