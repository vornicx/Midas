"""Learned-sparse lexical ranking (SPLADE++ / BM42) — a drop-in for BM25 in the hybrid RRF.

BM25 matches exact terms; learned sparse catches lexical-but-reweighted matches a bi-encoder ranks
low — SPLADE expands a term to related terms with learned weights, BM42 reweights BM25 with
transformer attention. It's the precision lever for dated facts, codes, and identifiers (the BEAM
`knowledge_update` / `temporal` categories). Same `.scores(query)` surface as `bm25.BM25`, so
`Memory`'s hybrid path stays backend-agnostic.

fastembed (ONNX, no torch) supplies the model; scoring is a sparse dot product over shared
dimensions, walked through an inverted index exactly like BM25's postings.

NOTE: unlike BM25 (pure-Python, instant rebuild), this embeds the whole corpus once per build —
fine for read-heavy / batch-loaded corpora (build once, query many, cached on the store's `version`
counter). A write-heavy production store would cache per-record sparse vectors and update them
incrementally; that optimization is out of scope for the retrieval-quality experiment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .types import MemoryRecord


class SparseLexicalIndex:
    """Inverted sparse index over a record set; `.scores(query)` → id → dot-product score.

    `model` is a fastembed `SparseTextEmbedding` (or any object exposing `.embed(texts)` /
    `.query_embed(text)` yielding objects with integer `.indices` and float `.values`).
    """

    def __init__(self, records: list[MemoryRecord], *, model: Any) -> None:
        self.records = records
        self._model = model
        # term-dimension -> [(doc index, weight)]; a query visits only docs sharing one of its dims.
        self._postings: dict[int, list[tuple[int, float]]] = {}
        texts = [r.content for r in records]
        if texts:
            for i, emb in enumerate(model.embed(texts)):
                for idx, val in zip(emb.indices.tolist(), emb.values.tolist()):
                    self._postings.setdefault(int(idx), []).append((i, float(val)))

    def scores(self, query: str) -> dict[str, float]:
        """Map record id -> sparse dot-product score (only records with a positive score)."""
        q = next(iter(self._model.query_embed(query)))
        by_doc: dict[int, float] = {}
        for idx, qval in zip(q.indices.tolist(), q.values.tolist()):
            postings = self._postings.get(int(idx))
            if not postings:
                continue
            qv = float(qval)
            for i, dval in postings:
                by_doc[i] = by_doc.get(i, 0.0) + qv * dval
        return {self.records[i].id: s for i, s in by_doc.items() if s > 0.0}


class SparseLexicalBackend:
    """Loads a fastembed sparse model once and builds a `SparseLexicalIndex` per corpus.

    Callable as `(records) -> SparseLexicalIndex`, matching the `BM25(records)` factory shape that
    `Memory(lexical_index_factory=...)` expects — so a sparse model swaps in without touching the
    hybrid fusion code.
    """

    def __init__(
        self,
        model_name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions",
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        from fastembed import SparseTextEmbedding  # lazy, optional dependency

        from .embeddings import configure_local_model_cache

        self.model_name = model_name
        self.cache_dir = configure_local_model_cache(cache_dir)
        self._model = SparseTextEmbedding(model_name=model_name, cache_dir=self.cache_dir)

    def __call__(self, records: list[MemoryRecord]) -> SparseLexicalIndex:
        return SparseLexicalIndex(records, model=self._model)
