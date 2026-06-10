"""Midas v0 — semantic agentic memory.

`recall()` ranks candidates by relevance x importance x recency. Relevance is semantic
cosine, or — when a `reranker` (cross-encoder) is configured — its sharper score. A
`min_relevance` floor enforces **parsimony**: low-value memories are dropped rather than
padded into the budget (padding hurts — the eval's budget sweep showed more context can
lower answer quality via distraction). `build_context()` optionally expands each hit with
its same-thread neighbours (`window`) and packs the highest-value memories into a budgeted,
prompt-ready block (critical context first, so it sits at the top of a prompt, not buried).

Supersession (belief revision) uses a dual approach:
1. Semantic similarity: high cosine similarity indicates related content
2. Update pattern detection: textual cues like "actually", "instead", "now", "has taken over"
   indicate a state change, triggering supersession even with lower semantic similarity.
"""
from __future__ import annotations

import datetime as _dt
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from .embeddings import Embedder, HashingEmbedder, cosine
from .importance import ContentImportance
from .policy import DEFAULT_POLICY, MemoryPolicy
from .store import InMemoryStore
from .types import MEMORY_PROVENANCE, MemoryKind, MemoryProvenance, MemoryRecord, RecallHit

RECENCY_HALF_LIFE_DAYS = 30.0
DEFAULT_RECALL_LIMIT = 6

# Temporal horizons for tiering (short -> medium -> long). Recency-based and orthogonal to
# durability: a memory's *tier* is how recent it is; whether it survives forgetting also depends on
# its kind/importance (the durable semantic tier is kept regardless of age). These thresholds name
# the "short / medium / multi-day" horizons; they are reporting/policy knobs, not magic numbers.
SHORT_TERM_DAYS = 1.0    # <= 1 day  : short-term, recent raw turns
MEDIUM_TERM_DAYS = 7.0   # <= 1 week : medium-term; older than this is long-term / multi-day
# Durable kinds = the semantic/identity tier (distilled facts, standing preferences, hard
# constraints). Never decayed away by forgetting — they are the point of long-horizon memory.
DURABLE_KINDS: tuple[MemoryKind, ...] = ("fact", "preference", "constraint")

# Patterns that indicate a belief update/change (case-insensitive)
_UPDATE_PATTERNS = [
    # Direct updates
    r"\bactually\b",
    r"\binstead\b",
    r"\bnow\b",
    r"\bcurrently\b",
    r"\bupdated\b",
    r"\bchanged\b",
    r"\bchange\b",
    r"\bnew\b",
    r"\breplaced\b",
    r"\breplace\b",
    # Leadership/ownership changes
    r"\btaken over\b",
    r"\btake over\b",
    r"\bhanded over\b",
    r"\bhand over\b",
    r"\bsigned off\b",
    r"\bresigned\b",
    r"\bstepped down\b",
    r"\bstepped up\b",
    # Time/date changes
    r"\bpostponed\b",
    r"\bdelayed\b",
    r"\bmoved to\b",
    r"\bmoved from\b",
    r"\brescheduled\b",
    r"\bcancelled\b",
    # Negations that indicate change
    r"\bno longer\b",
    r"\bnot anymore\b",
    r"\bnever\b",
    r"\bwas\b.*\bbut now\b",
]


# Compile patterns for efficiency
_UPDATE_RE = re.compile("|".join(_UPDATE_PATTERNS), re.IGNORECASE)

# STRICT revision cues for *conversational* belief revision. Plain chat rarely contains these, so
# gating chat-supersession on them avoids the catastrophic over-supersession of all-chat data (LoCoMo
# recall 0.62 -> 0.00) while still catching genuine "I changed X" updates. Deliberately excludes
# common words ("now", "new", "change", "never") that fire on ordinary chatter.
_CONVO_UPDATE_PATTERNS = [
    r"\bactually\b",
    r"\binstead\b",
    r"\bno longer\b",
    r"\bnot anymore\b",
    r"\banymore\b",
    r"\bused to\b",
    r"\bthese days\b",
    r"\bcurrently\b",
    r"\bnowadays\b",
    r"\bswitch(?:ed)?\s+(?:to|from|back)\b",
    r"\bchang(?:ed|ing)\s+(?:to|my|the|from|it)\b",
    r"\bmov(?:ed|ing)\s+(?:to|from)\b",
    r"\bupdat(?:ed|e\s+my|e\s+the)\b",
    r"\brenamed\b",
    r"\breplac(?:ed|ing)\b",
    r"\bnow\s+(?:i'?m|it'?s|we'?re|my|the|use|using|prefer|live|living|work|working|have)\b",
]
_CONVO_UPDATE_RE = re.compile("|".join(_CONVO_UPDATE_PATTERNS), re.IGNORECASE)


# Generic stopwords for the domain-agnostic content-word overlap that gates supersession. We keep
# NO synonym map on purpose: paraphrase detection ("go live" ~ "launch") is the embedder's job via
# cosine similarity, and a hand-tuned map only overfits one dataset (it previously hardcoded this
# benchmark's character names and month words). Content-word overlap is just a cheap topical anchor.
_SUPERSEDE_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'can', 'we', 'our', 'us', 'it', 'its',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'they', 'them',
}
_SUPERSEDE_ENTITY_STOPWORDS = {
    "project", "user", "client", "team", "ticket", "issue", "task", "feature", "release", "launch",
    "date", "database", "service", "server", "environment",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december",
}
_PROPER_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_-]{2,}\b")
_PROVENANCE_RANK = {
    "planning": 0,
    "observation": 1,
    "action": 2,
    "user_confirmation": 3,
}


def _content_words(text: str) -> set[str]:
    """Salient (non-stopword, >3-char, alphanumeric) lowercased words — a domain-agnostic topical
    fingerprint used only to gate medium-similarity supersession. No synonym expansion by design."""
    return {
        w.lower()
        for w in text.split()
        if len(w) > 3 and w.lower() not in _SUPERSEDE_STOPWORDS and w.isalnum()
    }


def _proper_entities(text: str) -> set[str]:
    """Very cheap named-entity guard for supersession.

    If two otherwise-similar memories are about different named things (Apollo vs Artemis), they are
    near-duplicates, not updates. We only use this as a negative guard; embeddings still do the recall.
    """
    return {
        w.lower()
        for w in _PROPER_ENTITY_RE.findall(text)
        if w.lower() not in _SUPERSEDE_ENTITY_STOPWORDS
    }


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[float]: ...


@dataclass
class ContextBlock:
    text: str
    records: list[MemoryRecord] = field(default_factory=list)
    tokens: int = 0


@dataclass
class CaptureResult:
    """Outcome of an auto-`capture`: whether Midas's policy kept the turn, and why/why not. Returned to
    the agent so it learns the relevance bar instead of guessing it."""

    stored: bool
    reason: str
    record: MemoryRecord | None = None
    importance: int = 0


class Memory:
    def __init__(
        self,
        *,
        store: InMemoryStore | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        calibration_reranker: Reranker | None = None,  # scores context for abstention; does NOT reorder
        now: Callable[[], float] | None = None,
        w_relevance: float = 1.0,
        w_importance: float = 0.3,
        w_recency: float = 0.2,
        supersede: bool = False,
        supersede_threshold: float = 0.55,
        supersede_pool: int = 20,
        supersede_margin: float = 0.05,
        supersede_lower_threshold: float = 0.30,
        supersede_same_kind_only: bool = True,
        supersede_conversational: bool = False,
        supersede_contradiction_threshold: float = 0.5,
        min_relevance_ratio: float = 0.3,  # drop hits < ratio*top-hit relevance (0 disables)
        exclude_superseded_from_context: bool = True,
        abstention_threshold: float = 0.3,
        abstention_relevance_floor: float = 0.0,  # >0: abstain when top-hit relevance is below this
        abstention_entailment_floor: float = 0.0,  # >0: abstain when no turn ENTAILS an answer (NLI)
        nli: object | None = None,  # LocalNLI: precise contradiction (revise) + entailment (abstain)
        importance_scorer: Callable[[str], int] | None = None,  # derive importance from content (no LLM)
        policy: MemoryPolicy | None = None,  # relevance parameters Midas imposes when `capture`-ing
        novelty_weight: float = 0.0,  # >0: blend novelty-vs-store into derived importance (no LLM)
        reinforce: bool = False,  # a restated memory gains importance + recency (repetition ⇒ salience)
        reinforce_threshold: float = 0.85,
        reinforce_bump: int = 1,
        include_provenance: bool = False,
    ) -> None:
        self.store = store if store is not None else InMemoryStore()
        self.embedder = embedder if embedder is not None else HashingEmbedder()
        self.reranker = reranker
        self._calibration_reranker = calibration_reranker
        self._now = now if now is not None else time.time
        self.w_relevance = w_relevance
        self.w_importance = w_importance
        self.w_recency = w_recency
        self._supersede = supersede
        self._supersede_threshold = supersede_threshold
        self._supersede_pool = supersede_pool
        self._supersede_margin = supersede_margin
        self._supersede_lower_threshold = supersede_lower_threshold
        self._supersede_same_kind_only = supersede_same_kind_only
        self._supersede_conversational = supersede_conversational
        self._supersede_contradiction_threshold = supersede_contradiction_threshold
        # Scale-free parsimony default. Measured (deterministic, all offline datasets + LongMemEval):
        # at 0.3 it never evicted a gold turn anywhere, while doubling precision@k and cutting ~30%
        # of context tokens on spread-scale embedders; 0.4+ starts evicting buried updates (multiday
        # recall 1.00 -> 0.80). On bge's compressed cosine scale 0.3 is a no-op — it only prunes
        # what is *far* below the query's own best hit.
        self._min_relevance_ratio = max(0.0, min_relevance_ratio or 0.0)
        self._nli = nli
        self._importance_scorer = importance_scorer
        self._policy = policy or DEFAULT_POLICY
        self._novelty_weight = max(0.0, min(1.0, novelty_weight))
        self._reinforce = reinforce
        self._reinforce_threshold = reinforce_threshold
        self._reinforce_bump = reinforce_bump
        self._exclude_superseded_from_context = exclude_superseded_from_context
        self._abstention_threshold = abstention_threshold
        self._abstention_relevance_floor = abstention_relevance_floor
        self._abstention_entailment_floor = abstention_entailment_floor
        self._include_provenance = include_provenance
        # Cached BM25 lexical index for hybrid recall, keyed by the store's change counter —
        # rebuilding it per query made hybrid ~35x slower at LongMemEval scale (66 ms vs 1.9 ms).
        self._bm25_cache: tuple[int, Any] | None = None

    def remember(
        self,
        content: str,
        *,
        kind: MemoryKind = "note",
        importance: int | None = None,
        source: str | None = None,
        provenance: MemoryProvenance = "observation",
        actor: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: float | None = None,
    ) -> MemoryRecord:
        content = (content or "").strip()
        if not content:
            raise ValueError("Memory.remember: `content` is required")
        embedding = _as_f32(self.embedder.embed(content))
        # `created_at` is the EVENT time (when the fact was true/stated), distinct from ingest order.
        # Passing it makes recency and chronological context reflect real time, not load order — the
        # bitemporal signal long-horizon temporal reasoning needs. Defaults to the wall clock.
        ts = created_at if created_at is not None else self._now()
        if self._reinforce:
            self._reinforce_existing(embedding, ts)  # a restatement boosts the matched memory
        importance = self._derive_importance(importance, content, embedding)
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            content=content,
            kind=kind,
            importance=importance,
            source=source,
            provenance=_normalize_provenance(provenance),
            actor=actor,
            metadata=metadata or {},
            created_at=ts,
            updated_at=ts,
            embedding=embedding,
        )
        self.store.put(record)
        if self._supersede:
            self._maybe_supersede(record)
        return record

    def remember_many(self, items: list[dict[str, Any]]) -> list[MemoryRecord]:
        """Remember multiple records, batching embeddings when the embedder supports it."""
        if not items:
            return []

        normalized: list[dict[str, Any]] = []
        contents: list[str] = []
        for item in items:
            content = (item.get("content") or "").strip()
            if not content:
                raise ValueError("Memory.remember_many: every item requires `content`")
            normalized.append({**item, "content": content})
            contents.append(content)

        embed_many = getattr(self.embedder, "embed_many", None)
        embeddings = (
            embed_many(contents)
            if callable(embed_many)
            else [self.embedder.embed(content) for content in contents]
        )
        if len(embeddings) != len(normalized):
            raise ValueError("Memory.remember_many: embedder returned the wrong number of embeddings")

        records: list[MemoryRecord] = []
        for item, embedding in zip(normalized, embeddings):
            embedding = _as_f32(embedding)  # store as float32 array, not list[float] (footprint)
            ts = item.get("created_at")  # event time when provided (see remember); else ingest time
            ts = ts if ts is not None else self._now()
            if self._reinforce:
                self._reinforce_existing(embedding, ts)  # a restatement boosts the matched memory
            record = MemoryRecord(
                id=str(uuid.uuid4()),
                content=item["content"],
                kind=item.get("kind", "note"),
                importance=self._derive_importance(item.get("importance"), item["content"], embedding),
                source=item.get("source"),
                provenance=_normalize_provenance(item.get("provenance") or "observation"),
                actor=item.get("actor"),
                metadata=item.get("metadata") or {},
                created_at=ts,
                updated_at=ts,
                embedding=embedding,
            )
            self.store.put(record)
            if self._supersede:
                self._maybe_supersede(record)
            records.append(record)
        return records

    def capture(
        self,
        content: str,
        *,
        kind: MemoryKind = "chat",
        source: str | None = None,
        provenance: MemoryProvenance = "observation",
        actor: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: float | None = None,
        policy: MemoryPolicy | None = None,
    ) -> CaptureResult:
        """Auto-remember a turn **only if it clears the relevance policy** (no LLM). Scores the content's
        importance (`ContentImportance`), then enforces the policy's floor + accepted-kinds + dedup, and
        reports the outcome. This is what makes "capture everything" safe: the agent forwards turns and
        *Midas decides* what is worth keeping (and says why). For unconditional storage, use `remember`."""
        content = (content or "").strip()
        if not content:
            return CaptureResult(False, "empty content")
        policy = policy or self._policy
        scorer = self._importance_scorer or ContentImportance()
        embedding = self.embedder.embed(content)
        importance = self._blend_novelty(scorer(content), embedding)
        if kind not in policy.accept_kinds:
            return CaptureResult(False, f"kind '{kind}' not accepted by policy", importance=importance)
        if importance < policy.min_importance:
            return CaptureResult(
                False,
                f"below relevance floor (importance {importance} < {policy.min_importance})",
                importance=importance,
            )
        if policy.dedup_threshold and policy.dedup_threshold > 0:
            near = self.store.search(embedding, limit=1)
            if near and near[0][0] >= policy.dedup_threshold:
                existing = near[0][1]
                upgraded = self._upgrade_provenance(
                    existing,
                    provenance=provenance,
                    actor=actor,
                    source=source,
                    metadata=metadata,
                    updated_at=created_at,
                )
                if self._reinforce:
                    self._reinforce_record(existing, created_at if created_at is not None else self._now())
                    return CaptureResult(
                        False, "reinforced an existing memory (restatement)", record=existing,
                        importance=importance,
                    )
                reason = "near-duplicate of an existing memory"
                if upgraded:
                    reason += " (provenance upgraded)"
                return CaptureResult(
                    False, reason, record=existing, importance=importance,
                )
        record = self.remember(
            content, kind=kind, importance=importance, source=source,
            provenance=provenance, actor=actor,
            metadata=metadata, created_at=created_at,
        )
        return CaptureResult(True, "stored", record=record, importance=importance)

    def _upgrade_provenance(
        self,
        record: MemoryRecord,
        *,
        provenance: MemoryProvenance,
        actor: str | None,
        source: str | None,
        metadata: dict[str, Any] | None,
        updated_at: float | None,
    ) -> bool:
        provenance = _normalize_provenance(provenance)
        if _PROVENANCE_RANK[provenance] <= _PROVENANCE_RANK[record.provenance]:
            return False
        record.provenance = provenance
        record.actor = actor or record.actor
        record.source = source or record.source
        record.metadata = {**record.metadata, **(metadata or {})}
        record.updated_at = updated_at if updated_at is not None else self._now()
        self.store.put(record)
        return True

    def _novelty(self, embedding: list[float] | None) -> float:
        """Novelty-vs-store: 1 - max cosine similarity to anything already stored (1 = wholly new,
        0 = an exact duplicate). A store-aware salience signal — a turn dissimilar to every existing
        memory carries new information; a near-restatement of a known fact does not. No LLM."""
        if embedding is None:
            return 1.0
        hits = self.store.search(embedding, limit=1)
        max_sim = max(0.0, hits[0][0]) if hits else 0.0
        return max(0.0, 1.0 - max_sim)

    def _reinforce_record(self, record: MemoryRecord, now: float) -> None:
        """A memory was just restated -> it matters more (importance climbs) and is current again
        (recency refreshes). The principled inverse of novelty: repetition signals salience."""
        record.importance = _clamp_importance(record.importance + self._reinforce_bump)
        record.updated_at = now

    def _reinforce_existing(self, embedding: list[float] | None, now: float) -> MemoryRecord | None:
        """When a new turn closely matches an existing memory (cosine >= `reinforce_threshold`), treat
        it as a restatement and reinforce that memory. Returns the reinforced record, or None. No LLM."""
        if not self._reinforce or embedding is None:
            return None
        hits = self.store.search(embedding, limit=1)
        if hits and hits[0][0] >= self._reinforce_threshold:
            self._reinforce_record(hits[0][1], now)
            return hits[0][1]
        return None

    def _blend_novelty(self, base_importance: int, embedding: list[float] | None) -> int:
        """Blend a content-salience importance (1-5) with novelty-vs-store via `novelty_weight` in
        [0,1] (a convex combination of the two normalised signals), then bucket back to 1-5. At weight
        0 this is exactly the content score; higher weights let a *new* fact outrank a *repeated* one
        even when both look equally salient in isolation — which is what the content heuristic alone
        cannot see."""
        base = _clamp_importance(base_importance)
        if self._novelty_weight <= 0 or embedding is None:
            return base
        nov = self._novelty(embedding)
        content_norm = (base - 1) / 4
        combined = (1 - self._novelty_weight) * content_norm + self._novelty_weight * nov
        return _clamp_importance(1 + round(combined * 4))

    def _derive_importance(
        self, importance: int | None, content: str, embedding: list[float] | None = None
    ) -> int:
        """Importance as given, else derived (no LLM): the content scorer's salience (or 1 when no
        scorer), optionally blended with novelty-vs-store. Used by `remember`/`remember_many` so raw
        turns get a salience without the caller annotating every one."""
        if importance is not None:
            return _clamp_importance(importance)
        base = self._importance_scorer(content) if self._importance_scorer else 1
        return self._blend_novelty(base, embedding)

    def recall(
        self,
        query: str,
        *,
        limit: int = DEFAULT_RECALL_LIMIT,
        kind: MemoryKind | None = None,
        min_importance: int | None = None,
        min_relevance: float | None = None,
        min_relevance_ratio: float | None = None,
        pool: int | None = None,
        hybrid: bool = False,
        fusion: str = "rrf",
        now: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RecallHit]:
        q_emb = self.embedder.embed(query)
        # `now` is the query's reference time (e.g. when the question is asked); recency decays
        # relative to it, so "what is my current X" favours the most recent evidence. Defaults to
        # the wall clock, which leaves time-agnostic datasets behaving exactly as before.
        now = now if now is not None else self._now()

        def predicate(record: MemoryRecord) -> bool:
            if kind is not None and record.kind != kind:
                return False
            if min_importance is not None and record.importance < min_importance:
                return False
            if metadata_filter:
                # Equality scoping (namespace / user / agent / project): every key must match.
                for key, value in metadata_filter.items():
                    if record.metadata.get(key) != value:
                        return False
            return True

        # Vector prefilter to a candidate pool, then score and (optionally) rerank.
        pool = pool or max(limit * 5, limit)
        relevance_map: dict[str, float] = {}
        if hybrid:
            candidates, relevance_map = self._hybrid_candidates(
                query, q_emb, pool=pool, predicate=predicate, fusion=fusion
            )
        else:
            candidates = self.store.search(q_emb, limit=pool, predicate=predicate)
        if self._supersede:
            candidates = self._resolve_candidates(candidates)
        if not candidates:
            return []

        if hybrid:
            # Fused lexical+semantic relevance (RRF by default; see _hybrid_candidates) — a lexical
            # match the bi-encoder ranked low still earns relevance and survives the top-k cut.
            # (No cross-encoder here — it reorders within-budget items but doesn't change recall.)
            relevances = [relevance_map.get(rec.id, max(0.0, cos)) for cos, rec in candidates]
        elif self.reranker is not None:
            # Sharper relevance: cross-encoder score (squashed to 0..1) over the pool.
            raw = self.reranker.rerank(query, [rec.content for _, rec in candidates])
            relevances = [max(_sigmoid(s), max(0.0, cos)) for s, (cos, _) in zip(raw, candidates)]
        else:
            relevances = [max(0.0, cos) for cos, _ in candidates]

        # Relative parsimony floor: drop hits far below the query's own best hit. Unlike the
        # absolute `min_relevance` (whose useful value depends on the embedder's cosine scale),
        # the ratio is scale-free, so one default transfers across embedders and queries.
        # None falls back to the instance default (0.3, measured-safe); pass 0 to disable.
        ratio = self._min_relevance_ratio if min_relevance_ratio is None else min_relevance_ratio
        ratio_floor = ratio * max(relevances) if ratio and relevances else None

        hits: list[RecallHit] = []
        for (_, record), relevance in zip(candidates, relevances):
            if min_relevance is not None and relevance < min_relevance:
                continue  # parsimony: drop low-value memories rather than pad the budget
            if ratio_floor is not None and relevance < ratio_floor:
                continue
            importance_norm = (record.importance - 1) / 4
            age_days = max(0.0, (now - record.updated_at) / 86_400.0)
            recency = math.exp(-age_days / RECENCY_HALF_LIFE_DAYS)
            score = (
                self.w_relevance * relevance
                + self.w_importance * importance_norm
                + self.w_recency * recency
            )
            hits.append(RecallHit(record, score, relevance, importance_norm, recency))
        hits.sort(key=lambda h: (h.score, h.record.updated_at), reverse=True)
        return hits[:limit]

    def _hybrid_candidates(
        self,
        query: str,
        q_emb: list[float],
        *,
        pool: int,
        predicate: Callable[[MemoryRecord], bool],
        fusion: str = "rrf",
    ) -> tuple[list[tuple[float, MemoryRecord]], dict[str, float]]:
        """Union the semantic top-`pool` with the BM25 top-`pool` (lexical), computing cosine for any
        BM25-only records, and return a fused relevance in [0,1] per record id. Pure lexical+vector
        fusion (no LLM) so lexically-relevant memories the embedder ranks low can still surface.

        fusion="rrf" (default): reciprocal-rank fusion, score = sum of 1/(k+rank) over the dense and
        lexical rankings — rank-based, so it sidesteps the cosine/BM25 scale mismatch that made the
        older fusion="max" (max of cosine and max-normalised BM25) displace true gold at tight budgets.
        """
        semantic = self.store.search(q_emb, limit=pool, predicate=predicate)
        records = [r for r in self.store.all() if predicate(r)]
        allowed = {r.id for r in records}
        # Score with the cached whole-store index, then keep only predicate-passing records. (IDF is
        # corpus-wide either way at the default no-filter recall; rebuilding a filtered index per
        # query was the 35x latency hole.)
        bm_scores = {
            rid: s for rid, s in self._bm25_index().scores(query).items() if rid in allowed
        }
        bm_ranked = sorted(bm_scores, key=bm_scores.get, reverse=True)

        by_id: dict[str, tuple[float, MemoryRecord]] = {rec.id: (cos, rec) for cos, rec in semantic}
        rec_by_id = {r.id: r for r in records}
        for rid in bm_ranked[:pool]:
            if rid not in by_id:
                rec = rec_by_id[rid]
                cos = cosine(q_emb, rec.embedding) if rec.embedding is not None else 0.0
                by_id[rid] = (cos, rec)

        if fusion == "rrf":
            k = 60  # standard RRF damping constant
            dense_rank = {rec.id: i for i, (_, rec) in enumerate(semantic)}
            bm_rank = {rid: i for i, rid in enumerate(bm_ranked)}
            raw = {
                rid: (1.0 / (k + dense_rank[rid] + 1) if rid in dense_rank else 0.0)
                + (1.0 / (k + bm_rank[rid] + 1) if rid in bm_rank else 0.0)
                for rid in by_id
            }
            top = max(raw.values(), default=0.0) or 1.0
            relevance_map = {rid: s / top for rid, s in raw.items()}
        else:  # "max": legacy max(cosine, max-normalised BM25)
            max_bm = max(bm_scores.values(), default=0.0) or 1.0
            relevance_map = {
                rid: max(max(0.0, by_id[rid][0]), bm_scores.get(rid, 0.0) / max_bm) for rid in by_id
            }
        return list(by_id.values()), relevance_map

    def _bm25_index(self):
        """The BM25 index over the whole store, rebuilt only when the store has changed (keyed by
        the store's `version` counter). Falls back to a per-call build for custom stores without
        a counter."""
        from .bm25 import BM25

        records = self.store.all()  # for SQLiteStore this also refreshes from other processes
        version = getattr(self.store, "version", None)
        if version is None:
            return BM25(records)
        if self._bm25_cache is None or self._bm25_cache[0] != version:
            self._bm25_cache = (version, BM25(records))
        return self._bm25_cache[1]

    def recall_confidence(self, query: str, **recall_kwargs: Any) -> tuple[list[RecallHit], float]:
        """Recall with confidence score.
        
        Returns the recall hits along with a confidence score (0-1) indicating
        how confident the system is that the retrieved memories answer the query.
        Lower confidence suggests the system should abstain from answering.
        """
        hits = self.recall(query, **recall_kwargs)
        if not hits:
            return hits, 0.0
        
        # Confidence is based on the top hit's score (normalized 0-1)
        # A score above the abstention threshold means we're confident
        top_score = hits[0].score
        # Normalize score to 0-1 range based on maximum possible score
        max_possible = self.w_relevance * 1.0 + self.w_importance * 1.0 + self.w_recency * 1.0
        confidence = min(1.0, max(0.0, top_score / max_possible))
        return hits, confidence

    def guard_reliance(
        self,
        query: str,
        *,
        intended_use: str = "planning",
        acting_agent: str | None = None,
        limit: int = DEFAULT_RECALL_LIMIT,
        **recall_kwargs: Any,
    ):
        """Recall evidence for `query` and decide whether it can justify `intended_use`.

        This is the Guard boundary: memory that is fine for planning may still be insufficient for an
        external action unless it came from explicit user confirmation.
        """
        from .guard import decide_memory_use

        hits = self.recall(query, limit=limit, **recall_kwargs)
        return decide_memory_use(
            [h.record for h in hits],
            intended_use=intended_use,
            acting_agent=acting_agent,
        )

    def _maybe_supersede(self, new_record: MemoryRecord) -> None:
        """LLM-free belief revision: when a new memory expresses the same belief as an existing one
        (high embedding similarity, same kind), mark the old head superseded so recall returns only
        the current value.

        Paraphrased updates ("we'll go live in October" replacing "launch is September 14") are
        caught by the *embedder*, not a keyword map: an explicit change cue (_UPDATE_RE: "now",
        "actually", "instead", "taken over"...) lowers the similarity bar, and a domain-agnostic
        content-word overlap gates the medium band so a merely-related belief is not revised away.
        Restricted to durable kinds (never chatter), single best head only (with an ambiguity
        guard). Approximate by design and measured against the multi-day detector; see the doc.
        """
        if new_record.embedding is None:
            return
        is_chat = new_record.kind == "chat"
        if is_chat and self._supersede_same_kind_only:
            # Conversational belief revision (opt-in): a chat turn may revise an earlier chat belief
            # ONLY if it carries a STRICT revision cue ("actually", "no longer", "switched to"...).
            # Ordinary chatter has no cue -> stays inert, preserving conversational recall (guardrail
            # against the LoCoMo 0.62->0.00 over-supersession). Without the opt-in, chat never revises.
            if not (self._supersede_conversational and _CONVO_UPDATE_RE.search(new_record.content)):
                return

        # An explicit update cue lowers the bar: real revisions are often paraphrased (lower cosine)
        # yet signal the change lexically. With no cue, demand strong similarity to revise a belief.
        has_update_pattern = bool(_UPDATE_RE.search(new_record.content))
        effective_threshold = (
            self._supersede_lower_threshold if has_update_pattern else self._supersede_threshold
        )
        if is_chat:
            # Conversations are noisy and embeddings cluster by topic, so demand STRONG similarity
            # (not the lowered cue-threshold) before revising a chat belief — precision over recall.
            effective_threshold = max(effective_threshold, self._supersede_threshold)

        predicate = lambda r: (
            r.id != new_record.id
            and (not self._supersede_same_kind_only or r.kind == new_record.kind)
        )
        candidates = self.store.search(
            new_record.embedding, limit=self._supersede_pool, predicate=predicate
        )

        by_head: dict[str, tuple[float, MemoryRecord]] = {}
        for score, candidate in candidates:
            if score < effective_threshold:
                continue
            head = self._resolve_head(candidate)
            if head.id == new_record.id or head.superseded_by is not None:
                continue
            if " ".join(head.content.split()).lower() == " ".join(new_record.content.split()).lower():
                continue  # exact duplicates are consolidation/dedup work, not belief revision
            new_entities = _proper_entities(new_record.content)
            head_entities = _proper_entities(head.content)
            if new_entities and head_entities and not (new_entities & head_entities):
                continue
            # Below the strong threshold the embedder isn't confident it's the SAME belief; require a
            # shared salient word as a cheap, domain-agnostic topical anchor (no synonym map). Chat is
            # noisy, so demand the topical anchor for it regardless of score.
            if (score < self._supersede_threshold or is_chat) and not (
                _content_words(new_record.content) & _content_words(head.content)
            ):
                continue
            if head.id not in by_head or score > by_head[head.id][0]:
                by_head[head.id] = (score, head)

        if not by_head:
            return
        ranked = sorted(by_head.values(), key=lambda item: item[0], reverse=True)
        # Ambiguity guard: if the two best heads are ~equally similar, don't guess which to revise.
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < self._supersede_margin:
            return
        head = ranked[0][1]
        # Precision gate (the principled no-LLM fix): with a local NLI model, only revise a belief the
        # new record actually CONTRADICTS — not merely one it's similar to. This is what makes
        # conversational revision safe on diverse data, where "similar + cue" is often a distinct fact.
        if self._nli is not None:
            if self._nli.contradiction(new_record.content, head.content) < self._supersede_contradiction_threshold:
                return
        head.superseded_by = new_record.id

    def _resolve_head(self, record: MemoryRecord) -> MemoryRecord:
        seen: set[str] = set()
        current = record
        while current.superseded_by is not None and current.id not in seen:
            seen.add(current.id)
            nxt = self.store.get(current.superseded_by)
            if nxt is None:
                break
            current = nxt
        return current

    def _resolve_candidates(
        self, candidates: list[tuple[float, MemoryRecord]]
    ) -> list[tuple[float, MemoryRecord]]:
        """Map each stale candidate to its current head (keeping the match score that
        surfaced the chain), de-duplicated by head — so a query phrased like the OLD value
        resolves to the CURRENT belief."""
        best: dict[str, tuple[float, MemoryRecord]] = {}
        for score, record in candidates:
            head = self._resolve_head(record)
            if head.id not in best or score > best[head.id][0]:
                best[head.id] = (score, head)
        return list(best.values())

    def build_context(
        self,
        query: str,
        *,
        token_budget: int = 512,
        limit: int = DEFAULT_RECALL_LIMIT,
        header: bool = True,
        window: int = 0,
        thread_key: str | None = None,
        neighbor_min_importance: int | None = None,
        max_record_chars: int = 600,
        include_abstention: bool = True,
        context_order: str = "relevance",
        include_provenance: bool | None = None,
        now: float | None = None,
        **recall_kwargs: Any,
    ) -> ContextBlock:
        hits = self.recall(query, limit=limit, now=now, **recall_kwargs)
        records = [h.record for h in hits]
        if window > 0 and thread_key is not None:
            records = self._append_neighbors(
                records,
                window=window,
                thread_key=thread_key,
                min_importance=neighbor_min_importance,
                # Neighbour expansion must respect the recall scope, or a same-thread record from
                # another namespace would leak past the filter.
                metadata_filter=recall_kwargs.get("metadata_filter"),
            )

        # Filter out superseded records if enabled
        if self._supersede and self._exclude_superseded_from_context:
            records = [
                r for r in records 
                if self._resolve_head(r).id == r.id  # Only keep current heads
            ]

        # Fill the budget in priority order (highest-value first). Build-context defaults to lean lines
        # because Memory(include_provenance=False) is now the default; full provenance/source remains one
        # `recall()`/`inspect_memory(id)` away, and callers can opt into audit-mode lines explicitly.
        use_provenance = self._include_provenance if include_provenance is None else include_provenance
        chosen: list[tuple[MemoryRecord, str, int]] = []
        used = 0
        for record in records:
            line = format_record(
                record,
                max_chars=max_record_chars,
                include_provenance=use_provenance,
                now=now,
            )
            if use_provenance and record.source:
                line = f"{line} [source: {record.source}]"
            cost = approx_tokens(line)
            if used + cost > token_budget:
                break
            chosen.append((record, line, cost))
            used += cost
        if not chosen:
            return ContextBlock(text="", records=[], tokens=0)

        if context_order == "chronological":
            chosen.sort(key=lambda item: item[0].created_at)
        elif context_order == "recency":
            chosen.sort(key=lambda item: item[0].created_at, reverse=True)
        elif context_order != "relevance":
            raise ValueError("context_order must be 'relevance', 'chronological', or 'recency'")
        chosen_records = [record for record, _, _ in chosen]
        body = "\n".join(line for _, line, _ in chosen)
        
        # Calibrated (abstention). The top hit's RELEVANCE is the calibration signal: with the
        # cross-encoder reranker on it cleanly separates answerable (~0.93) from unanswerable (~0.17)
        # — where the bi-encoder cosine cannot (0.70 vs 0.64). A low-relevance top hit means no stored
        # memory actually answers the query, so tell the reader to abstain instead of confabulating
        # from topically-near distractors (Midas's strong retrieval otherwise *tempts* a guess).
        if include_abstention and hits:
            # Calibration score: a SEPARATE cross-encoder pass over the IN-CONTEXT records (it does not
            # reorder retrieval — that side-effect sank the naive version). The cross-encoder, unlike
            # bi-encoder cosine, tells answer-match (~0.93) from mere topic-match (~0.17): a low score
            # means none of the surfaced memories actually answer, so warn the reader off guessing.
            # NLI entailment (the principled Calibrated signal): does ANY in-context turn *entail* an
            # answer to the question? Tests answer-PRESENCE directly — answerable median 0.98 vs
            # unanswerable 0.37, where cosine/relevance both fail (~0.7 vs ~0.64). Local, no LLM.
            ent = None
            if self._nli is not None and self._abstention_entailment_floor > 0 and chosen_records:
                hyp = f"This passage answers the question: {query}"
                ent = max((self._nli.entailment(r.content, hyp) for r in chosen_records), default=0.0)
            cal = None
            if self._calibration_reranker is not None and self._abstention_relevance_floor > 0 and chosen_records:
                raw = self._calibration_reranker.rerank(query, [r.content for r in chosen_records])
                cal = max((_sigmoid(s) for s in raw), default=0.0)
            top_conf = min(1.0, max(0.0, hits[0].score / (
                self.w_relevance + self.w_importance + self.w_recency)))
            if ent is not None and ent < self._abstention_entailment_floor:
                # No turn entails an answer -> drop the tempting snippets so the only honest reply is
                # "I don't know" (a soft caution backfires: the reader scrutinises and confabulates).
                body = "(No stored memory answers this question.)"
            elif cal is not None and cal < self._abstention_relevance_floor:
                body = "(No stored memory is relevant to this question.)"
            elif self._abstention_threshold > 0 and top_conf < self._abstention_threshold:
                body = body.rstrip() + "\n\n[Memory: confidence low - consider abstaining if unsure]"

        if header:
            head = "# Memory - relevant context (use silently to stay consistent)"
            # Anchor the reader's clock so it can resolve relative time ("how long ago...") against
            # the dated memories — the signal LongMemEval temporal-reasoning questions need.
            if now is not None:
                head += f"\n# Today is {_fmt_date(now)}"
            body = head + "\n" + body
        return ContextBlock(text=body, records=chosen_records, tokens=used)

    def _expand_neighbors(
        self,
        hits: list[RecallHit],
        *,
        window: int,
        thread_key: str,
        min_importance: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        """Expand each hit with its ±window same-thread neighbours, in hit-priority order.
        Threads are grouped by `record.metadata[thread_key]`, ordered by insertion."""
        threads: dict[Any, list[MemoryRecord]] = {}
        for record in self.store.all():
            if metadata_filter and any(
                record.metadata.get(k) != v for k, v in metadata_filter.items()
            ):
                continue
            threads.setdefault(record.metadata.get(thread_key), []).append(record)
        index: dict[str, tuple[Any, int]] = {}
        for key, recs in threads.items():
            for i, rec in enumerate(recs):
                index[rec.id] = (key, i)

        ordered: list[MemoryRecord] = []
        seen: set[str] = set()
        for hit in hits:
            rec = hit.record
            located = index.get(rec.id)
            if located is None:
                if rec.id not in seen:
                    seen.add(rec.id)
                    ordered.append(rec)
                continue
            key, idx = located
            recs = threads[key]
            for j in range(max(0, idx - window), min(len(recs), idx + window + 1)):
                nb = recs[j]
                if self._supersede and nb.superseded_by is not None:
                    continue
                if min_importance is not None and nb.importance < min_importance:
                    continue
                if nb.id not in seen:
                    seen.add(nb.id)
                    ordered.append(nb)
        return ordered

    def _append_neighbors(
        self,
        records: list[MemoryRecord],
        *,
        window: int,
        thread_key: str,
        min_importance: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        """Keep direct hits first, then append same-thread neighbours in hit-priority order."""
        direct_ids = {record.id for record in records}
        ordered = list(records)
        seen = set(direct_ids)
        for nb in self._expand_neighbors(
            [RecallHit(record, 0.0, 0.0, 0.0, 0.0) for record in records],
            window=window,
            thread_key=thread_key,
            min_importance=min_importance,
            metadata_filter=metadata_filter,
        ):
            if nb.id in seen or nb.id in direct_ids:
                continue
            seen.add(nb.id)
            ordered.append(nb)
        return ordered

    def memory_value(self, record: MemoryRecord, *, now: float | None = None) -> float:
        """Decay-based retention value in [0, 1] — how worth keeping a memory is, independent of any
        query. It blends the two query-agnostic signals `recall()` already ranks by — **importance**
        and **recency** — and is what `forget_decayed()` evicts the bottom of. (Relevance is excluded:
        it depends on a query; forgetting is a query-free maintenance pass.) High = keep; low = a
        forgetting candidate (old AND unimportant). No LLM."""
        now = now if now is not None else self._now()
        importance_norm = (record.importance - 1) / 4
        age_days = max(0.0, (now - record.updated_at) / 86_400.0)
        recency = math.exp(-age_days / RECENCY_HALF_LIFE_DAYS)
        denom = self.w_importance + self.w_recency
        if denom <= 0:
            return 0.0
        return (self.w_importance * importance_norm + self.w_recency * recency) / denom

    def tier(self, record: MemoryRecord, *, now: float | None = None) -> str:
        """The memory's temporal horizon by age: 'short' (<= 1d), 'medium' (<= 1w), or 'long'
        (older / multi-day). This names the short/medium/multi-day tiers explicitly. Orthogonal to
        durability — `forget_decayed` keeps the durable semantic tier (DURABLE_KINDS / high
        importance) even when its age puts it in 'long'."""
        now = now if now is not None else self._now()
        age_days = max(0.0, (now - record.updated_at) / 86_400.0)
        if age_days <= SHORT_TERM_DAYS:
            return "short"
        if age_days <= MEDIUM_TERM_DAYS:
            return "medium"
        return "long"

    def forget_decayed(
        self,
        *,
        now: float | None = None,
        max_records: int | None = None,
        min_value: float | None = None,
        protect_importance: int = 4,
        protect_kinds: tuple[MemoryKind, ...] = DURABLE_KINDS,
        keep_chains: bool = True,
    ) -> list[str]:
        """Selective forgetting (no LLM): bound memory growth by evicting the lowest-`memory_value`
        records — the aging, low-importance, non-durable long tail — so storage and the retrieved
        context stay small as an agent runs for days/weeks. This is the maintenance half of
        long-horizon memory: 2026 evidence is that *selective* forgetting beats both unlimited
        retention (storage cost + retrieval dilution; the budget sweep showed volume hurts answers)
        and blind compression (loses the critical detail). It is also the basis of the time tiers:
        recent and durable memories keep a high value; the medium/long tail decays out.

        **Protections — never forgotten** (the long-term semantic/identity tier is sacred):
          - kind in `protect_kinds` (durable: facts, preferences, constraints);
          - importance >= `protect_importance` (critical);
          - with `keep_chains` (default), any record in a supersession chain — it has `superseded_by`
            set, or another record points to it. Dropping one mid-chain would orphan `_resolve_head`
            (a query phrased like the old value would resolve to a deleted record). Compacting stale
            chains is a separate, careful operation.

        **Policy:** if `min_value` is set, drop every unprotected record scoring below it; if
        `max_records` is set, additionally drop the lowest-value unprotected records until the store
        holds <= `max_records`. Returns the forgotten ids, lowest-value first — the deletion audit
        trail (what was dropped and in what order), which the enterprise retention/“right to be
        forgotten” story needs. A no-op (returns []) when nothing is unprotected or both limits are
        None."""
        now = now if now is not None else self._now()
        records = self.store.all()
        # Anything still referenced by a `superseded_by` link must not be orphaned.
        pointed_to = {r.superseded_by for r in records if r.superseded_by is not None}

        def protected(r: MemoryRecord) -> bool:
            if r.kind in protect_kinds or r.importance >= protect_importance:
                return True
            if keep_chains and (r.superseded_by is not None or r.id in pointed_to):
                return True
            return False

        candidates = [r for r in records if not protected(r)]
        value = {r.id: self.memory_value(r, now=now) for r in candidates}
        scored = sorted(candidates, key=lambda r: value[r.id])  # lowest value first

        to_drop: list[str] = []
        dropped: set[str] = set()
        if min_value is not None:
            for r in scored:
                if value[r.id] < min_value:
                    to_drop.append(r.id)
                    dropped.add(r.id)
        if max_records is not None:
            surviving = len(records) - len(dropped)
            for r in scored:
                if surviving <= max_records:
                    break
                if r.id in dropped:
                    continue
                to_drop.append(r.id)
                dropped.add(r.id)
                surviving -= 1

        return [rid for rid in to_drop if self.store.delete(rid)]

    def forget(self, record_id: str) -> bool:
        """Delete one memory by id, repairing any supersession chain that runs through it.

        A record mid-chain may be the `superseded_by` target of older records; deleting it naively
        would make `_resolve_head` dead-end on a missing id (a query phrased like the old value
        resolves to nothing). Relinking those pointers to the deleted record's own successor keeps
        every chain walkable. Explicit deletion ignores the forgetting protections on purpose —
        "forget this" from a user outranks retention policy."""
        target = self.store.get(record_id)
        if target is None:
            return False
        for record in self.store.all():
            if record.superseded_by == record_id:
                record.superseded_by = target.superseded_by
                self.store.put(record)
        return self.store.delete(record_id)

    def forget_matching(
        self,
        query: str,
        *,
        min_relevance: float = 0.5,
        limit: int = 20,
        kind: MemoryKind | None = None,
        metadata_filter: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> list[MemoryRecord]:
        """Topic-level erasure (right-to-be-forgotten): delete every memory matching `query` at
        relevance >= `min_relevance`, returning the matched records as the audit trail.

        With `dry_run=True` nothing is deleted — the returned records are a reviewable preview, the
        safe default for agent-driven erasure. Deletion uses `forget`, so supersession chains stay
        walkable, and it bypasses the durability protections: an explicit "remove what you know
        about X" must also remove protected facts about X. No LLM."""
        hits = self.recall(
            query,
            limit=limit,
            kind=kind,
            min_relevance=min_relevance,
            metadata_filter=metadata_filter,
        )
        matched = [h.record for h in hits if h.relevance >= min_relevance]
        if not dry_run:
            for record in matched:
                self.forget(record.id)
        return matched

    def consolidate(
        self,
        *,
        similarity_threshold: float = 0.95,
        now: float | None = None,
        keep_chains: bool = True,
        pool: int = 50,
    ) -> list[str]:
        """Extractive consolidation (no LLM): collapse near-**duplicate** memories — restatements of the
        *same* thing — keeping the single highest-value copy and dropping the redundant rest. This is the
        compression half of the medium/long tiers: a long conversation restates the same fact many times
        ("still targeting Sept 14", "launch is Sept 14", "Sept 14 is launch day"), which inflates storage
        and dilutes retrieval with copies. Distinct from supersession (which revises a *contradicted*
        belief) — this removes *redundancy*, not staleness.

        Extractive, not abstractive: it drops whole duplicate records rather than LLM-summarise them, so
        the zero-LLM-at-ingest wedge holds and every surviving memory keeps its real source/provenance.
        Greedy + conservative (the supersession lesson — over-merging destroys recall): representatives
        are chosen **highest-value first**, only cosine **>= `similarity_threshold`** (very near) is
        merged, and supersession-chain members are left intact (`keep_chains`). Returns the dropped ids
        (audit trail). Run as a periodic maintenance pass; one pass merges clusters up to `pool` wide."""
        now = now if now is not None else self._now()
        records = [r for r in self.store.all() if r.embedding is not None]
        pointed_to = {r.superseded_by for r in records if r.superseded_by is not None}

        def in_chain(r: MemoryRecord) -> bool:
            return r.superseded_by is not None or r.id in pointed_to

        # Highest-value first: the copy we keep as the cluster representative.
        records.sort(key=lambda r: self.memory_value(r, now=now), reverse=True)
        dropped: set[str] = set()
        for rep in records:
            if rep.id in dropped or (keep_chains and in_chain(rep)):
                continue
            near = self.store.search(
                rep.embedding,
                limit=pool,
                predicate=lambda r: r.id != rep.id and r.id not in dropped,
            )
            for sim, other in near:
                if sim < similarity_threshold:
                    break  # search is sorted by similarity desc
                if keep_chains and in_chain(other):
                    continue
                dropped.add(other.id)

        return [d for d in dropped if self.store.delete(d)]

    def assemble(self, query: str, **kwargs: Any) -> str:
        """Prompt-ready context string (thin wrapper over `build_context`)."""
        return self.build_context(query, **kwargs).text


def _as_f32(embedding):
    """Store the embedding as a float32 numpy array, not a Python list[float]. Embeddings dominate the
    in-memory footprint and a list of floats costs ~32 B/value (PyObject + pointer) vs 4 B for float32
    — a ~7x reduction at 768 dims. Falls back to the list unchanged when numpy is absent. (SQLite
    persistence already stores float32 BLOBs; this fixes the in-memory side.)"""
    if embedding is None:
        return None
    try:
        import numpy as np
    except ImportError:
        return embedding
    return np.asarray(embedding, dtype=np.float32)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _clamp_importance(value: Any) -> int:
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        return 1
    return max(1, min(5, n))


def _normalize_provenance(value: str) -> MemoryProvenance:
    if value in MEMORY_PROVENANCE:
        return value  # type: ignore[return-value]
    allowed = ", ".join(MEMORY_PROVENANCE)
    raise ValueError(f"invalid memory provenance '{value}' (expected one of: {allowed})")


def _fmt_date(ts: float) -> str:
    """Render an epoch as a UTC date. UTC (not local) so event dates match the dataset's own
    timestamps and relative-time answers ("how many days ago") don't drift by a timezone offset."""
    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).strftime("%Y-%m-%d")


def _relative_age(created_at: float, now: float) -> str:
    """Human relative age ("3w ago", "today") so a reader can answer "how long ago…" by reading,
    not by date subtraction — which a non-reasoning reader often gets wrong."""
    days = int((now - created_at) // 86_400)
    if days <= 0:
        return "today"
    if days < 14:
        return f"{days}d ago"
    if days < 60:
        return f"{days // 7}w ago"
    if days < 730:
        return f"{days // 30}mo ago"
    return f"{days // 365}y ago"


def format_record(
    record: MemoryRecord,
    *,
    max_chars: int = 600,
    include_provenance: bool = False,
    now: float | None = None,
) -> str:
    # Show the event date, plus its age relative to `now` when known (the temporal-reasoning anchor).
    date = _fmt_date(record.created_at)
    if now is not None:
        date = f"{date}, {_relative_age(record.created_at, now)}"
    body = " ".join(record.content.split())[:max_chars]
    if include_provenance:
        actor = f" | actor:{record.actor}" if record.actor else ""
        return (
            f"- [{record.kind}/{record.provenance} | imp{record.importance} | {date} "
            f"| id:{record.id[:8]}{actor}] {body}"
        )
    # Default line stays lean: importance is Midas's internal ranking signal, noise to the reader —
    # dropping it makes budget room for the date/age the reader actually uses.
    return f"- [{record.kind} | {date}] {body}"


def approx_tokens(text: str) -> int:
    """Crude token estimate (~1.3 tokens/word). Install tiktoken for exact counts."""
    return max(1, int(len(text.split()) * 1.3))
