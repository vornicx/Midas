"""Retrieval query-adapter experiment (eval-only, no torch, no LLM) — the precision lever.

Learn a single linear map on the QUERY embedding (not the index, not the embedding model) so
retrieval aligns better with what queries need.
It acts ONLY on the query, costs nothing at ingest, and needs no reindex — the one shape of retrieval
improvement that fits Midas's no-LLM, source-traceable bet (a reranker/LLM-extractor does not).

This is a MEASURED experiment, never a shipped default. The rule (see BENCHMARKS / frontier-2026, where
every retrieval *component* lever so far failed to generalize): ship only if it improves recall on a
HELD-OUT split across datasets AND doesn't regress provenance/decision categories. We expect little —
and a clean negative is a perfectly good result.

Adapter: W (d x d), initialized to the identity. Triplet hinge loss on L2-normalized embeddings,
  L = max(0, margin - (W q)·p + (W q)·n),
trained by SGD with an L2 pull toward identity, so the adapter degrades gracefully to the baseline
unless the data genuinely moves it. Evaluated on held-out queries: recall@k of baseline (q·turn) vs
adapted ((W q)·turn). numpy only — torch stays out of Midas entirely, even here.

Run:  uv run python -m eval.retrieval_adapter --convs 3 --k 10
"""
from __future__ import annotations

import argparse
import random

import numpy as np

from midas import LocalEmbedder

from .datasets import beam


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def train_adapter(
    triplets: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    dim: int,
    *,
    epochs: int = 60,
    lr: float = 0.1,
    margin: float = 0.2,
    l2: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """SGD on the triplet hinge loss. W starts at the identity (= baseline) and the l2 term pulls it
    back there, so the adapter only departs from baseline where the triplets consistently demand it.
    Gradient of (W q)·v w.r.t. W is outer(v, q), so the active-margin gradient is outer(n - p, q)."""
    W = np.eye(dim, dtype=np.float64)
    eye = np.eye(dim, dtype=np.float64)
    rng = random.Random(seed)
    order = list(range(len(triplets)))
    for _ in range(epochs):
        rng.shuffle(order)
        for i in order:
            q, p, n = triplets[i]
            Wq = W @ q
            grad = l2 * (W - eye)
            if margin - Wq @ p + Wq @ n > 0:  # margin violated -> push toward p, away from n
                grad = grad + np.outer(n - p, q)
            W -= lr * grad
    return W


def _recall_at_k(query_rows: list[tuple[set[int], np.ndarray]], k: int) -> float:
    """query_rows: (gold_row_indices, scores_over_corpus). Mean fraction of gold in the top-k."""
    recalls = []
    for gold, scores in query_rows:
        topk = set(np.argsort(-scores)[:k].tolist())
        recalls.append(len(gold & topk) / len(gold))
    return float(np.mean(recalls)) if recalls else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="100K")
    ap.add_argument("--convs", type=int, default=3)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--l2", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    embedder = LocalEmbedder()
    dataset = beam(tier=args.tier, max_conversations=args.convs)

    # Per conversation: embed turns once; collect each answerable question with its gold rows + corpus.
    samples: list[dict] = []  # {qy, gold:set[int], E:(m,d)}
    dim = 0
    for sample in dataset.samples:
        ids = [e.id for e in sample.events]
        row_of = {eid: i for i, eid in enumerate(ids)}
        E = _normalize(np.array([embedder.embed(e.content) for e in sample.events], dtype=np.float64))
        dim = E.shape[1]
        for q in sample.questions:
            gold = {row_of[g] for g in q.gold_event_ids if g in row_of}
            if not gold:
                continue
            qy = _normalize(np.array(embedder.embed(q.text), dtype=np.float64))
            samples.append({"qy": qy, "gold": gold, "E": E})

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    n_test = max(1, int(len(samples) * args.test_frac))
    test, train = samples[:n_test], samples[n_test:]

    # Triplets from TRAIN only: per gold positive, the hardest non-gold turn (max baseline score) as neg.
    triplets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for s in train:
        scores = s["E"] @ s["qy"]
        non_gold = [i for i in range(s["E"].shape[0]) if i not in s["gold"]]
        if not non_gold:
            continue
        neg = non_gold[int(np.argmax(scores[non_gold]))]
        for g in s["gold"]:
            triplets.append((s["qy"], s["E"][g], s["E"][neg]))

    W = train_adapter(triplets, dim, epochs=args.epochs, l2=args.l2, seed=args.seed)

    def _eval(split, adapt: bool):
        rows = []
        for s in split:
            q = (W @ s["qy"]) if adapt else s["qy"]
            rows.append((s["gold"], s["E"] @ q))
        return _recall_at_k(rows, args.k)

    base_test, adpt_test = _eval(test, False), _eval(test, True)
    base_tr, adpt_tr = _eval(train, False), _eval(train, True)

    print(f"\n=== query-adapter on BEAM-{args.tier}: recall@{args.k} "
          f"(dim={dim}, {len(train)} train / {len(test)} test queries, {len(triplets)} triplets) ===")
    print(f"{'split':<8}{'baseline':>10}{'adapted':>10}{'Δ':>8}")
    print(f"{'train':<8}{base_tr:>10.3f}{adpt_tr:>10.3f}{adpt_tr - base_tr:>+8.3f}")
    print(f"{'test':<8}{base_test:>10.3f}{adpt_test:>10.3f}{adpt_test - base_test:>+8.3f}")
    print("\nVerdict: ship only if TEST improves (train-only gains = overfit). "
          "Re-run across datasets before believing it.")


if __name__ == "__main__":
    main()
