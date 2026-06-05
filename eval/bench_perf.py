"""Performance + footprint benchmark — the latency / throughput / memory numbers the project never
measured (the exec report had to delete *invented* latency figures; this measures the real ones).

Everything here is local and deterministic — **no LLM, no API**. The only model is the embedder (bge),
which is the honest cost of "no LLM at ingest": you still run an embedding forward pass, just not an
extraction LLM. We report it separately so the trade-off is visible.

    uv run --no-sync python -m eval.bench_perf --local            # bge (real)
    uv run --no-sync python -m eval.bench_perf                    # hashing (offline floor)
"""
from __future__ import annotations

import argparse
import statistics
import time
import tracemalloc


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    i = min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))
    return xs[i]


def _ms(x: float) -> str:
    return f"{x * 1000:.2f} ms"


def main() -> None:
    ap = argparse.ArgumentParser(description="Midas latency / throughput / footprint benchmark")
    ap.add_argument("--local", action="store_true", help="use the local bge embedder (real); else hashing")
    ap.add_argument("--n", type=int, default=2000, help="records to ingest")
    ap.add_argument("--q", type=int, default=200, help="queries to time")
    args = ap.parse_args()

    from midas import Memory

    if args.local:
        from midas.embeddings import LocalEmbedder

        embedder = LocalEmbedder()
        label = f"local:{embedder.model_name}"
    else:
        from midas import HashingEmbedder

        embedder = HashingEmbedder()
        label = "hashing (offline)"

    mem = Memory(embedder=embedder)
    embedder.embed("warm up the model so it is excluded from the timings")  # exclude one-time load

    # Realistic, varied content (decisions/facts an agent would store).
    records = [
        f"Decision {i}: route service-{i % 50} through region eu-{i % 5} with a {i % 30 + 1}s timeout "
        f"because the {['payments', 'search', 'auth', 'ingest', 'billing'][i % 5]} team owns it."
        for i in range(args.n)
    ]
    queries = [f"timeout for service-{i % 50} in payments" for i in range(args.q)]

    # --- ingest (embed + store), with REAL footprint via tracemalloc (not a formula — the previous
    # estimate badly undercounted: a Python list of floats is ~32 B/float, not 8) ---
    embed_times, ingest_times = [], []
    tracemalloc.start()
    mem_before = tracemalloc.get_traced_memory()[0]
    t0 = time.perf_counter()
    for r in records:
        s = time.perf_counter()
        embedder.embed(r)  # measure the embed component on its own (the model cost)
        embed_times.append(time.perf_counter() - s)
        s = time.perf_counter()
        mem.remember(r)
        ingest_times.append(time.perf_counter() - s)
    ingest_wall = time.perf_counter() - t0
    mem_after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    # --- query (embed query + scan + assemble) ---
    query_times = []
    for q in queries:
        s = time.perf_counter()
        mem.build_context(q, token_budget=512, limit=8)
        query_times.append(time.perf_counter() - s)

    # --- footprint (REAL, measured by tracemalloc — not a formula) ---
    recs = mem.store.all()
    dim = len(recs[0].embedding) if recs and recs[0].embedding is not None else 0
    bytes_per_record = (mem_after - mem_before) / max(1, len(recs))
    mb_per_1k = bytes_per_record * 1000 / 1024 / 1024
    list_mb_per_1k = (dim * 32 + 400) * 1000 / 1024 / 1024  # what list[float] storage would have cost

    print(f"\nMidas perf — embedder: {label}  |  ingested: {args.n}  |  queries: {args.q}  |  dim: {dim}")
    print("LLM calls: 0   ·   API cost: $0   ·   data egress: 0 bytes   (structural — no extraction LLM)\n")
    print("| metric | p50 | p95 | mean |")
    print("| --- | --- | --- | --- |")
    print(f"| embed 1 record | {_ms(_pct(embed_times,50))} | {_ms(_pct(embed_times,95))} | {_ms(statistics.mean(embed_times))} |")
    print(f"| remember 1 record (embed+store) | {_ms(_pct(ingest_times,50))} | {_ms(_pct(ingest_times,95))} | {_ms(statistics.mean(ingest_times))} |")
    print(f"| build_context 1 query | {_ms(_pct(query_times,50))} | {_ms(_pct(query_times,95))} | {_ms(statistics.mean(query_times))} |")
    print(f"\nThroughput: ingest {args.n / ingest_wall:,.0f} rec/s (single; batched remember_many is "
          f"far faster)  ·  query {args.q / sum(query_times):,.0f} q/s")
    print(f"Footprint (real, tracemalloc): ~{bytes_per_record:,.0f} B/record  ·  ~{mb_per_1k:.1f} MB per "
          f"1,000 records (embeddings stored as float32 arrays; a list[float] would cost "
          f"~{list_mb_per_1k:.1f} MB/1k — ~{list_mb_per_1k / mb_per_1k:.0f}x more).")
    print(
        "\nCost note (honest): an LLM-ingest baseline (Mem0/Zep-style) runs an extraction prompt per "
        "turn — order ~500 in + ~100 out tokens/turn. Midas runs 0. We report token counts, not a "
        "fabricated $ figure: at 1M turns that is ~600M tokens you do NOT spend (price by your model)."
    )


if __name__ == "__main__":
    main()
