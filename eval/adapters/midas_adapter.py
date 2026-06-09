from __future__ import annotations

import os

from midas import Embedder, Memory

from ..schema import Event
from .base import RetrievalResult

_VALID_KINDS = {"note", "chat", "mission", "fact", "preference", "constraint"}


class MidasAdapter:
    """The wedge under test — now a thin wrapper over the midas SDK. The whole pipeline
    lives in the library: semantic recall over a pool -> cross-encoder rerank -> ±window
    same-session neighbour expansion -> budgeted assembly. No LLM at ingest/query (Midas's
    cost edge over Mem0's LLM extraction)."""

    name = "midas"
    uses_internal_llm = False  # no LLM at ingest/query - local embeddings only

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        limit: int = 8,
        neighbor_window: int = 1,
        rerank: bool = True,
        pool: int = 40,
        min_relevance: float | None = None,
        neighbor_min_importance: int | None = None,
        max_record_chars: int = 600,
        supersede: bool = False,
        supersede_threshold: float = 0.55,
        supersede_lower_threshold: float = 0.30,
        supersede_same_kind_only: bool = True,  # SAFE default: chat never supersedes chat
        supersede_conversational: bool = False,  # opt-in: chat revises chat on a strict cue
        use_nli: bool = False,  # local NLI: contradiction-gated revision (+ entailment abstention)
        # (False caused a catastrophic LoCoMo regression: all-chat data → mass over-supersession →
        #  recall@k 0.62 -> 0.00. Belief revision must be restricted to typed durable facts.)
        exclude_superseded_from_context: bool = True,
        abstention_threshold: float = 0.3,
        abstention_relevance_floor: float = 0.0,
        abstention_entailment_floor: float = 0.0,
        include_provenance: bool = False,
        context_order: str = "recency",
        hybrid: bool = False,
        hybrid_fusion: str = "rrf",
        time_aware: bool = True,  # use dataset event time for recency/chronology (vs ingest order)
        importance_scorer=None,  # no-LLM ContentImportance: derive importance from raw turns
        novelty_weight: float = 0.0,  # >0: blend novelty-vs-store into derived importance
        reinforce: bool = False,  # restated memories gain importance (repetition ⇒ salience)
    ) -> None:
        self._embedder = embedder
        self._limit = limit
        self._window = neighbor_window
        self._rerank = rerank and embedder is not None and os.getenv("MIDAS_RERANK", "1") != "0"
        self._pool = pool
        self._min_relevance = min_relevance
        self._neighbor_min_importance = neighbor_min_importance
        self._max_record_chars = max_record_chars
        self._supersede = supersede
        self._supersede_threshold = supersede_threshold
        self._supersede_lower_threshold = supersede_lower_threshold
        self._supersede_same_kind_only = supersede_same_kind_only
        self._supersede_conversational = supersede_conversational
        self._use_nli = use_nli
        self._nli = None  # cached across resets so the model loads once
        self._exclude_superseded_from_context = exclude_superseded_from_context
        self._abstention_threshold = abstention_threshold
        self._abstention_relevance_floor = abstention_relevance_floor
        self._abstention_entailment_floor = abstention_entailment_floor
        self._include_provenance = include_provenance
        self._context_order = context_order
        self._hybrid = hybrid
        self._hybrid_fusion = hybrid_fusion
        self._time_aware = time_aware
        self._importance_scorer = importance_scorer
        self._novelty_weight = novelty_weight
        self._reinforce = reinforce
        self._reranker = None  # cached across resets so the model loads once
        self.reset()

    def _get_nli(self):
        if (self._use_nli or self._abstention_entailment_floor > 0) and self._nli is None:
            from midas.nli import LocalNLI

            self._nli = LocalNLI()
        return self._nli

    def _get_reranker(self):
        # Loaded once and shared. Needed when retrieval reranks OR when abstention calibration does.
        if (self._rerank or self._abstention_relevance_floor > 0) and self._reranker is None:
            from midas import LocalReranker

            self._reranker = LocalReranker()
        return self._reranker

    def reset(self) -> None:
        reranker = self._get_reranker()
        self._mem = Memory(
            embedder=self._embedder,
            reranker=reranker if self._rerank else None,  # retrieval rerank only when enabled
            # Calibration reranker scores context for abstention WITHOUT reordering retrieval:
            calibration_reranker=reranker if self._abstention_relevance_floor > 0 else None,
            supersede=self._supersede,
            supersede_threshold=self._supersede_threshold,
            supersede_lower_threshold=self._supersede_lower_threshold,
            supersede_same_kind_only=self._supersede_same_kind_only,
            supersede_conversational=self._supersede_conversational,
            nli=self._get_nli(),
            importance_scorer=self._importance_scorer,
            novelty_weight=self._novelty_weight,
            reinforce=self._reinforce,
            exclude_superseded_from_context=self._exclude_superseded_from_context,
            abstention_threshold=self._abstention_threshold,
            abstention_relevance_floor=self._abstention_relevance_floor,
            abstention_entailment_floor=self._abstention_entailment_floor,
            include_provenance=self._include_provenance,
        )

    def ingest(self, events: list[Event]) -> None:
        items = []
        for event in events:
            kind = event.kind if event.kind in _VALID_KINDS else "note"
            items.append(
                {
                    "content": event.content,
                    "kind": kind,
                    # When a scorer or novelty weighting is configured, omit importance so Memory
                    # derives it (content salience + novelty-vs-store, no LLM); else honour the dataset.
                    "importance": None
                    if (self._importance_scorer is not None or self._novelty_weight > 0)
                    else event.importance,
                    "created_at": event.metadata.get("timestamp") if self._time_aware else None,
                    "provenance": event.metadata.get("provenance", "observation"),
                    "actor": event.metadata.get("actor"),
                    "metadata": {
                        "event_id": event.id,
                        "session": event.metadata.get("session") or event.metadata.get("session_id"),
                    },
                }
            )
        self._mem.remember_many(items)

    @property
    def store_size(self) -> int:
        """Number of records currently held — what selective forgetting bounds."""
        return len(self._mem.store.all())

    def forget_decayed(self, **kwargs) -> list[str]:
        """Run the SDK's no-LLM selective forgetting (decay-based eviction). Returns forgotten ids."""
        return self._mem.forget_decayed(**kwargs)

    def consolidate(self, **kwargs) -> list[str]:
        """Run the SDK's no-LLM extractive consolidation (dedup near-duplicates). Returns dropped ids."""
        return self._mem.consolidate(**kwargs)

    def query(self, question: str, *, token_budget: int, now: float | None = None) -> RetrievalResult:
        block = self._mem.build_context(
            question,
            token_budget=token_budget,
            limit=self._limit,
            window=self._window,
            thread_key="session",
            neighbor_min_importance=self._neighbor_min_importance,
            max_record_chars=self._max_record_chars,
            context_order=self._context_order,
            pool=self._pool,
            min_relevance=self._min_relevance,
            hybrid=self._hybrid,
            fusion=self._hybrid_fusion,
            now=now if self._time_aware else None,  # recency decays from when the question is asked
        )
        ids = [r.metadata.get("event_id") for r in block.records if r.metadata.get("event_id")]
        # Top turns by RELEVANCE (recall order, not the context's recency order) — the source an
        # answer was drawn from, for the NLI verifier. Cheap: embeddings are cached.
        hits = self._mem.recall(
            question, limit=self._limit, now=now if self._time_aware else None,
            pool=self._pool, min_relevance=self._min_relevance,
            hybrid=self._hybrid, fusion=self._hybrid_fusion,
        )
        top_texts = [h.record.content for h in hits]
        return RetrievalResult(
            context=block.text, retrieved_event_ids=ids, tokens=block.tokens, top_texts=top_texts
        )
