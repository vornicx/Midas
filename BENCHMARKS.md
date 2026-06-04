# Midas Benchmarks

Honest, reproducible benchmarks for the Midas agentic-memory SDK. Every number here comes from a
real run with the command to reproduce it. We deliberately **lead with reader-independent metrics**
(retrieval + cost) and treat end-to-end answer correctness as a secondary, noisy signal — see
*Methodology* for why that is the honest choice, not a convenient one.

## TL;DR

Midas isolates and wins the two axes that actually measure a *memory layer* (as opposed to the
reader LLM stacked on top):

- **Retrieval** — on LongMemEval-`s` (evidence buried among distractors), Midas retrieves the
  supporting turns at **recall@k 0.95** vs a recency-window baseline's **0.03**.
- **Cost** — Midas does **0 LLM calls, $0 API spend, and 0 data egress at ingest** (local embeddings
  only), versus LLM-at-ingest memory systems that call an LLM per session to extract facts.

## 1. Retrieval quality — `recall@k` (deterministic)

Fraction of gold supporting turns retrieved into the context. Fully deterministic (local embeddings,
no LLM), so it reproduces exactly.

| dataset | setting | baseline-raw | **Midas** |
|---|---|---:|---:|
| **LongMemEval-`s`** (buried evidence, hard retrieval) | n=40, bge-base, no rerank, seed 0 | 0.03 | **0.95** |
| **LoCoMo** (5 conversations) | n=50, bge-base, no rerank, seed 0 | 0.02 | **0.85** |

Across **both** datasets Midas retrieves the supporting turns at **0.85–0.95** while a recency window
gets **≤0.03** — the wedge holds beyond a single benchmark. On LongMemEval-`s` (n=40) the per-category
recall@k is strong across the board: fact 0.89 · multi-session 0.97 · knowledge-update 1.00 ·
temporal 0.95 · preference 1.00. A recency window finds essentially **none** of the buried evidence;
Midas finds ~9 in 10 — exactly the multi-session setting where retrieval quality decides whether the
answer is even *possible*. (`min_relevance` parsimony is a separate cost/quality knob; the numbers
above are pure retrieval, no pruning.)

**Time-aware retrieval (LLM-free).** Memories carry real **event time** (parsed from the dataset's
session timestamps), so recency and chronological context reflect *when things happened*, not load
order — the bitemporal signal long-horizon memory needs. Turning it on lifts **temporal recall@k
0.86 → 0.95** (n=40, A/B via `--midas-no-time`), in line with the LongMemEval paper's +7–11% for
temporal handling — but done with regex/relative-date math, **not** an LLM, preserving the no-LLM
ingest/query edge. (fact dips 0.92 → 0.89, within n=13 noise and with no effect on fact *answer*.)

```bash
# reproduce (deterministic; downloads LongMemEval-s on first run)
python -m eval.runner --dataset longmemeval --variant s --local \
  --local-max-text-chars 600 --local-batch-size 16 --midas-no-rerank \
  --max-questions 15 --limit 20 --seed 0
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

## 5. Correctness with a fixed strong reader (secondary)

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
the LLM-ingest systems offer — and its retrieval **scales to corpora where OM's keep-every-observation-
in-context design overflows** (LongMemEval-`m` = ~500 sessions). By category Midas **leads
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
layer**:

- Holding the reader fixed, a memory layer's lift is real; but swapping in a bigger reader moves the
  *headline* far more than the memory does. (Public SOTA on LongMemEval reports ~39% → ~83% from the
  memory system but ~83% → ~91% from *just a larger reader* — most of the headline is the reader.)
- Our own hosted LLM judge (an MoE served via API) is **not reproducible across sessions**: identical
  inputs scored ~0.46 one day and ~0.13 the next, even at temperature 0. We added a local,
  seed-pinned, serialized judge (`--judge-ollama`) to make correctness reproducible, but a small local
  reader is too weak to *use* good context — so correctness still does not cleanly isolate memory
  quality.

Therefore: **`recall@k` (deterministic, reader-independent) and ingest cost (structural) are our
primary metrics.** We report correctness only with a fixed reader and wide error bars, and never as a
headline.

### Honest caveats
- **Sample** is n=40 on LongMemEval-`s` and n=50 across 5 LoCoMo conversations. `recall@k` is
  deterministic, so the sample is real; the full LongMemEval set / all 10 LoCoMo conversations would
  tighten it further.
- **Latency is hardware/provider-dependent** (the ~668 ms for the LLM-at-ingest class includes API
  round-trip). The durable, hardware-independent claim is the **0-LLM / $0 / no-egress** column.
- **baseline-raw** = "stuff recent turns into the window" (the naive big-context approach).
- Numbers measured on CPU with `BAAI/bge-base-en-v1.5`. GPU / a faster embedder lowers Midas latency.
- **Reranking is off by default on large haystacks.** A cross-encoder reranker is available, but on
  LongMemEval-`s` (CPU) it added ~80× query latency (4.2 s vs 53 ms) with **no `recall@k` change**
  (0.88 → 0.88): it reorders the records that already fit the budget (which can help the *reader*) but
  does not change *which* evidence fits. So it is not on the retrieval-quality path here.

*All commands run from the repo root. `recall@k` requires no API key; `--judge*` flags do.*
