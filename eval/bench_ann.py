"""Benchmark the IVF ANN index vs the exact scan on REAL bge-base embeddings.

Uses the 36k real 768-d embeddings cached during the LongMemEval/LoCoMo runs (real semantic cluster
structure -- the regime IVF is designed for; uniform-random vectors would be IVF's worst case). Shows
(1) exact O(N) vs IVF sub-linear latency as N grows, (2) the recall@10 / latency trade-off vs nprobe.

    python -m eval.bench_ann
"""
from __future__ import annotations

import os
import sqlite3
import time

import numpy as np

from midas.ann import IVFIndex

DB = os.environ.get("MIDAS_EMBED_CACHE", r"D:\hf-cache\midas-embeddings.sqlite3")
N_QUERIES = 500
K = 10


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def load_vectors() -> tuple[np.ndarray, str]:
    """Real cached bge-base embeddings if available (real cluster structure), else synthetic
    clustered vectors (random/uniform vectors would be IVF's adversarial worst case, so we still
    plant clusters to stay representative of how embeddings actually distribute)."""
    if os.path.exists(DB):
        con = sqlite3.connect(DB)
        rows = con.execute("select embedding from embeddings where dim=768").fetchall()
        con.close()
        if rows:
            x = np.frombuffer(b"".join(r[0] for r in rows), dtype=np.float32).reshape(len(rows), 768)
            return _normalize(x.copy()), f"real bge-base 768-d (cache: {len(rows):,})"
    rng = np.random.default_rng(0)
    n, d, k = 36_000, 768, 600
    centers = _normalize(rng.normal(size=(k, d)).astype(np.float32))
    x = centers[rng.integers(0, k, n)] + 0.35 * rng.normal(size=(n, d)).astype(np.float32)
    return _normalize(x), f"synthetic clustered 768-d ({n:,}, {k} clusters)"


def exact_topk(corpus: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    sims = corpus @ q
    part = np.argpartition(-sims, k - 1)[:k]
    return part[np.argsort(-sims[part])]


def time_per_query(fn, queries: np.ndarray) -> float:
    fn(queries[0])  # warm up
    t0 = time.perf_counter()
    for q in queries:
        fn(q)
    return (time.perf_counter() - t0) / len(queries) * 1000.0  # ms/query


def recall_at_k(corpus: np.ndarray, queries: np.ndarray, index: IVFIndex, nprobe: int) -> float:
    hits = 0
    for q in queries:
        truth = set(exact_topk(corpus, q, K).tolist())
        got, _ = index.search(q, k=K, nprobe=nprobe)
        hits += len(truth & set(got.tolist()))
    return hits / (len(queries) * K)


def main() -> None:
    x, source = load_vectors()
    print(f"Loaded {len(x):,} vectors [{source}]  (queries held out: {N_QUERIES})\n")
    queries, corpus_full = x[:N_QUERIES], x[N_QUERIES:]

    # (1) Scaling curve: exact (O(N)) vs IVF (nprobe=8) as N grows.
    print("### Scaling: exact vs IVF (nprobe=8), real embeddings, k=10")
    print("| N corpus | nlist | exact ms/q | IVF ms/q | speedup | IVF recall@10 | cand scanned |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    sizes = [n for n in (5_000, 10_000, 20_000, len(corpus_full)) if n <= len(corpus_full)]
    for n in sizes:
        corpus = corpus_full[:n]
        nlist = max(1, int(round(np.sqrt(n))))
        index = IVFIndex(nlist=nlist, seed=0).fit(corpus)
        avg_cand = float(np.mean([len(index._lists[c]) for c in range(nlist)])) * 8
        exact_ms = time_per_query(lambda q: exact_topk(corpus, q, K), queries)
        ivf_ms = time_per_query(lambda q: index.search(q, k=K, nprobe=8), queries)
        rec = recall_at_k(corpus, queries, index, nprobe=8)
        print(f"| {n:,} | {nlist} | {exact_ms:.2f} | {ivf_ms:.3f} | {exact_ms/ivf_ms:.0f}x "
              f"| {rec:.3f} | ~{avg_cand:,.0f} |")

    # (2) Recall / latency vs nprobe at full N.
    corpus = corpus_full
    nlist = max(1, int(round(np.sqrt(len(corpus)))))
    index = IVFIndex(nlist=nlist, seed=0).fit(corpus)
    exact_ms = time_per_query(lambda q: exact_topk(corpus, q, K), queries)
    print(f"\n### Recall/latency vs nprobe  (N={len(corpus):,}, nlist={nlist}, exact={exact_ms:.2f} ms/q)")
    print("| nprobe | recall@10 | IVF ms/q | speedup |")
    print("|---:|---:|---:|---:|")
    for nprobe in (1, 2, 4, 8, 16, 32):
        rec = recall_at_k(corpus, queries, index, nprobe=nprobe)
        ivf_ms = time_per_query(lambda q: index.search(q, k=K, nprobe=nprobe), queries)
        print(f"| {nprobe} | {rec:.3f} | {ivf_ms:.3f} | {exact_ms/ivf_ms:.0f}x |")


if __name__ == "__main__":
    main()
