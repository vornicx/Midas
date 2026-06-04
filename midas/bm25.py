"""BM25 lexical ranking — pure-Python, zero-dep.

Fused with semantic recall (see `Memory.recall(hybrid=True)`) to catch lexically-relevant memories
the bi-encoder misses — a no-LLM lever to push retrieval past the embedding-only ceiling. Standard
Okapi BM25 with non-negative IDF; rebuilt per recall over the candidate set (cheap at eval scale).
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
        self._docs = [tokenize(r.content) for r in records]
        self._tfs = [Counter(d) for d in self._docs]
        self._dls = [len(d) for d in self._docs]
        self._avgdl = (sum(self._dls) / len(self._dls)) if self._dls else 0.0
        df: Counter = Counter()
        for doc in self._docs:
            for term in set(doc):
                df[term] += 1
        n = len(records)
        # Non-negative IDF (the BM25+ form) so common terms never push scores negative.
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> dict[str, float]:
        """Map record id -> BM25 score (only records with a positive score are included)."""
        q_terms = tokenize(query)
        avgdl = self._avgdl or 1.0
        out: dict[str, float] = {}
        for i, record in enumerate(self.records):
            tf = self._tfs[i]
            dl = self._dls[i]
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = self._idf.get(term, 0.0)
                s += idf * (f * (_K1 + 1)) / (f + _K1 * (1 - _B + _B * dl / avgdl))
            if s > 0.0:
                out[record.id] = s
        return out
