"""ColBERT late-interaction reranking via fastembed (ONNX, no torch).

A bi-encoder scores a query and a document each as ONE vector; ColBERT keeps a vector per TOKEN and
scores by MaxSim — for every query token, the best-matching document token, summed. That token-level
matching is sharper on exact entities (dates, codes, identifiers) than bi-encoder cosine, and
cheaper than a full cross-encoder (the document is encoded independently of the query; only the
small MaxSim is per-pair). Same `.rerank(query, docs) -> list[float]` surface as `LocalReranker`,
so it drops into `Memory(reranker=...)` unchanged.

Scores are returned as LOGITS of the length-normalised MaxSim, so the recall layer's `sigmoid()`
recovers a [0,1] relevance directly comparable to the bi-encoder cosine it is fused with — without
that calibration the raw MaxSim sum would saturate the sigmoid and erase the ranking signal.
"""
from __future__ import annotations

import math
from pathlib import Path


class ColBERTReranker:
    name = "colbert"

    def __init__(
        self,
        model_name: str = "answerdotai/answerai-colbert-small-v1",
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        from fastembed import LateInteractionTextEmbedding  # lazy, optional dependency

        from .embeddings import configure_local_model_cache

        self.cache_dir = configure_local_model_cache(cache_dir)
        self._model = LateInteractionTextEmbedding(model_name=model_name, cache_dir=self.cache_dir)
        self.model_name = model_name

    # Mirror LocalReranker's defensive length caps (an over-long pair blows the ONNX context).
    _MAX_QUERY_CHARS = 400
    _MAX_DOC_CHARS = 1200

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """MaxSim relevance per document, returned as a logit of the length-normalised score.
        Degrades to neutral scores on any runtime error rather than crashing the whole query."""
        if not documents:
            return []
        import numpy as np

        q = query[: self._MAX_QUERY_CHARS]
        docs = [d[: self._MAX_DOC_CHARS] for d in documents]
        try:
            q_emb = np.asarray(next(iter(self._model.query_embed(q))), dtype=np.float32)  # (nq, dim)
            nq = max(1, q_emb.shape[0])
            out: list[float] = []
            for d_emb in self._model.embed(docs):
                d = np.asarray(d_emb, dtype=np.float32)  # (nd, dim)
                if d.ndim != 2 or d.shape[0] == 0:
                    out.append(0.0)
                    continue
                # MaxSim: for each query token, its best doc-token dot product, averaged over query
                # tokens so the scale is ~[0,1] (answerai-colbert uses normalised token embeddings).
                maxsim = float((q_emb @ d.T).max(axis=1).sum()) / nq
                p = min(1.0 - 1e-6, max(1e-6, maxsim))
                out.append(math.log(p / (1.0 - p)))  # logit → recall's sigmoid recovers ~maxsim
            return out
        except Exception:
            return [0.0] * len(documents)
