"""BM25 lexical ranking — pure-Python, zero-dep.

Fused with semantic recall (see `Memory.recall(hybrid=True)`) to catch lexically-relevant memories
the bi-encoder misses — a no-LLM lever to push retrieval past the embedding-only ceiling. Standard
Okapi BM25 with non-negative IDF. The index is built over the whole store and cached by `Memory`
(keyed on the store's change counter), and scoring walks per-term posting lists, so a query touches
only the documents that share a term with it — not the whole corpus.
"""
from __future__ import annotations

import math
from collections import Counter

from .embeddings import tokenize
from .types import MemoryRecord

_K1 = 1.5
_B = 0.75


class BM25:
    def __init__(self, records: list[MemoryRecord]) -> None:
        self.records = records
        self._dls: list[int] = []
        # term -> [(doc index, term frequency)] — scoring visits only docs containing a query term.
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for i, record in enumerate(records):
            doc = tokenize(record.content)
            self._dls.append(len(doc))
            for term, f in Counter(doc).items():
                self._postings.setdefault(term, []).append((i, f))
        self._avgdl = (sum(self._dls) / len(self._dls)) if self._dls else 0.0
        n = len(records)
        # Non-negative IDF (the BM25+ form) so common terms never push scores negative.
        self._idf = {
            t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self._postings.items()
        }

    def scores(self, query: str) -> dict[str, float]:
        """Map record id -> BM25 score (only records with a positive score are included)."""
        q_terms = tokenize(query)
        avgdl = self._avgdl or 1.0
        by_doc: dict[int, float] = {}
        for term in q_terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for i, f in postings:
                dl = self._dls[i]
                by_doc[i] = by_doc.get(i, 0.0) + idf * (f * (_K1 + 1)) / (
                    f + _K1 * (1 - _B + _B * dl / avgdl)
                )
        return {self.records[i].id: s for i, s in by_doc.items() if s > 0.0}
