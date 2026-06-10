from __future__ import annotations

import math

from eval.adapters.midas_adapter import MidasAdapter
from eval.schema import Event
from midas import Memory


class FixedEmbedder:
    dim = 2

    def embed(self, text: str) -> list[float]:
        vectors = {
            "old launch": [1.0, 0.0],
            "new launch": [1.0, 0.0],
            "database": [0.0, 1.0],
            "query database": [0.0, 1.0],
        }
        return vectors[text]


class ChainEmbedder:
    dim = 2

    def embed(self, text: str) -> list[float]:
        y = math.sqrt(1 - 0.91**2)
        vectors = {
            "old belief": [1.0, 0.0],
            "current belief": [0.91, y],
            "new update": [0.91, -y],
        }
        return vectors[text]


class AmbiguousEmbedder:
    dim = 2

    def embed(self, text: str) -> list[float]:
        vectors = {
            "belief a": [1.0, 0.0],
            "belief b": [0.8, 0.6],
            "ambiguous update": [0.9486832980505138, 0.31622776601683794],
        }
        return vectors[text]


class AdversarialNearDuplicateEmbedder:
    dim = 2

    def embed(self, text: str) -> list[float]:
        vectors = {
            "Project Apollo launch moved to October 1": [1.0, 0.0],
            "Project Artemis launch moved to October 1": [1.0, 0.0],
            "Project Apollo launch moved to November 1": [1.0, 0.0],
        }
        return vectors[text]


class CountingBatchEmbedder(FixedEmbedder):
    def __init__(self) -> None:
        self.single_calls = 0
        self.batch_calls = 0

    def embed(self, text: str) -> list[float]:
        self.single_calls += 1
        return super().embed(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls += 1
        return [FixedEmbedder.embed(self, text) for text in texts]


class NeighborPackingEmbedder:
    dim = 2

    def embed(self, text: str) -> list[float]:
        vectors = {
            "query": [1.0, 0.0],
            "hit one": [1.0, 0.0],
            "neighbor one": [0.0, 1.0],
            "hit two": [0.9, 0.0],
        }
        return vectors[text]


def test_neighbor_expansion_drops_superseded_records() -> None:
    mem = Memory(embedder=FixedEmbedder(), supersede=True, supersede_threshold=0.9)

    mem.remember("old launch", kind="fact", metadata={"session": "day1"})
    mem.remember("database", kind="constraint", metadata={"session": "day1"})
    mem.remember("new launch", kind="fact", metadata={"session": "day2"})

    context = mem.assemble(
        "query database",
        token_budget=100,
        limit=1,
        window=1,
        thread_key="session",
    )

    assert "database" in context
    assert "old launch" not in context
    assert "new launch" not in context


def test_neighbor_expansion_can_filter_low_importance_records() -> None:
    mem = Memory(embedder=FixedEmbedder(), supersede=False)

    mem.remember("old launch", kind="chat", importance=1, metadata={"session": "day1"})
    mem.remember("database", kind="constraint", importance=5, metadata={"session": "day1"})

    context = mem.assemble(
        "query database",
        token_budget=100,
        limit=1,
        window=1,
        thread_key="session",
        neighbor_min_importance=2,
    )

    assert "database" in context
    assert "old launch" not in context


def test_remember_many_batches_embeddings() -> None:
    embedder = CountingBatchEmbedder()
    mem = Memory(embedder=embedder)

    records = mem.remember_many(
        [
            {"content": "old launch", "kind": "fact"},
            {"content": "database", "kind": "constraint"},
        ]
    )

    assert len(records) == 2
    assert embedder.batch_calls == 1
    assert embedder.single_calls == 0


def test_context_can_limit_record_body_length() -> None:
    mem = Memory()
    mem.remember("alpha beta gamma delta epsilon", kind="fact")

    context = mem.assemble("alpha", token_budget=100, max_record_chars=10)

    assert "alpha beta" in context
    assert "gamma" not in context


def test_build_context_can_omit_provenance_for_lean_reader_prompts() -> None:
    mem = Memory(abstention_threshold=0.0)
    mem.remember(
        "Decision: use PostgreSQL for the primary database.",
        kind="constraint",
        importance=5,
        source="mcp:test-session",
        provenance="user_confirmation",
        actor="user",
    )

    lean = mem.build_context("primary database", token_budget=100)
    audit = mem.build_context("primary database", token_budget=100, include_provenance=True)

    assert "PostgreSQL" in lean.text
    assert "id:" not in lean.text
    assert "source:" not in lean.text
    assert lean.tokens < audit.tokens
    assert "id:" in audit.text
    assert "source: mcp:test-session" in audit.text


def test_context_packs_direct_hits_before_neighbors() -> None:
    mem = Memory(embedder=NeighborPackingEmbedder(), abstention_threshold=0.0)
    mem.remember("hit one", kind="fact", metadata={"session": "s1"})
    mem.remember("neighbor one", kind="fact", metadata={"session": "s1"})
    mem.remember("hit two", kind="fact", metadata={"session": "s2"})

    context = mem.assemble(
        "query",
        token_budget=15,  # two lean lines (~7 tokens each) fit; the neighbour must not
        limit=2,
        window=1,
        thread_key="session",
    )

    assert "hit one" in context
    assert "hit two" in context
    assert "neighbor one" not in context


def test_context_defaults_to_relevance_order() -> None:
    # ratio=0: this test checks ORDERING, so keep the low-relevance record in instead of
    # letting the default parsimony floor prune it.
    mem = Memory(embedder=FixedEmbedder(), min_relevance_ratio=0)

    mem.remember("old launch", kind="fact")
    mem.remember("database", kind="constraint")

    context = mem.assemble("query database", token_budget=100, limit=2)

    assert context.index("database") < context.index("old launch")


def test_context_can_order_records_by_recency() -> None:
    times = iter([1.0, 2.0, 3.0])
    mem = Memory(  # ratio=0: ordering test — don't let parsimony prune the weaker record
        embedder=FixedEmbedder(), now=lambda: next(times), abstention_threshold=0.0,
        min_relevance_ratio=0,
    )

    mem.remember("old launch", kind="fact")
    mem.remember("database", kind="constraint")

    context = mem.assemble("query database", token_budget=100, limit=2, context_order="recency")

    assert context.index("database") < context.index("old launch")


def test_midas_adapter_omits_provenance_noise_by_default() -> None:
    adapter = MidasAdapter(
        embedder=FixedEmbedder(),
        limit=1,
        neighbor_window=0,
        rerank=False,
    )
    adapter.ingest([Event("db-event", "database", kind="constraint")])

    result = adapter.query("query database", token_budget=100)

    assert "database" in result.context
    assert "id:" not in result.context
    assert result.retrieved_event_ids == ["db-event"]


def test_recall_old_belief_resolves_to_current_head() -> None:
    mem = Memory(embedder=FixedEmbedder(), supersede=True, supersede_threshold=0.9)

    old = mem.remember("old launch", kind="fact")
    new = mem.remember("new launch", kind="fact")

    hits = mem.recall("old launch", limit=1)

    assert old.superseded_by == new.id
    assert [h.record.id for h in hits] == [new.id]


def test_chat_records_do_not_supersede_by_default() -> None:
    mem = Memory(embedder=FixedEmbedder(), supersede=True, supersede_threshold=0.9)

    old = mem.remember("old launch", kind="chat")
    mem.remember("new launch", kind="chat")

    assert old.superseded_by is None


def test_new_update_can_match_stale_chain_member_and_supersede_current_head() -> None:
    mem = Memory(embedder=ChainEmbedder(), supersede=True, supersede_threshold=0.9)

    old = mem.remember("old belief", kind="fact")
    current = mem.remember("current belief", kind="fact")
    new = mem.remember("new update", kind="fact")

    assert old.superseded_by == current.id
    assert current.superseded_by == new.id
    assert mem.recall("old belief", limit=1)[0].record.id == new.id


def test_ambiguous_update_does_not_supersede_any_belief() -> None:
    mem = Memory(
        embedder=AmbiguousEmbedder(),
        supersede=True,
        supersede_threshold=0.9,
        supersede_margin=0.05,
    )

    a = mem.remember("belief a", kind="fact")
    b = mem.remember("belief b", kind="fact")
    mem.remember("ambiguous update", kind="fact")

    assert a.superseded_by is None
    assert b.superseded_by is None


def test_adversarial_near_duplicate_different_entity_does_not_supersede() -> None:
    mem = Memory(
        embedder=AdversarialNearDuplicateEmbedder(),
        supersede=True,
        supersede_threshold=0.9,
    )

    apollo = mem.remember("Project Apollo launch moved to October 1", kind="fact")
    artemis = mem.remember("Project Artemis launch moved to October 1", kind="fact")

    assert apollo.superseded_by is None
    assert artemis.superseded_by is None


def test_temporal_conflict_same_entity_supersedes_old_belief() -> None:
    mem = Memory(
        embedder=AdversarialNearDuplicateEmbedder(),
        supersede=True,
        supersede_threshold=0.9,
    )

    old = mem.remember("Project Apollo launch moved to October 1", kind="fact")
    new = mem.remember("Project Apollo launch moved to November 1", kind="fact")

    assert old.superseded_by == new.id


class ConvoUpdateEmbedder:
    """Two 'main language' chat turns sit at cos≈0.95; the no-cue restatement is identical."""

    dim = 2

    def embed(self, text: str) -> list[float]:
        y = math.sqrt(1 - 0.95**2)
        vectors = {
            "user: my main programming language is python": [1.0, 0.0],
            "user: i actually switched my main language to rust now": [0.95, y],
            "user: my main language is still python": [1.0, 0.0],
        }
        return vectors[text]


def test_conversational_update_revises_chat_belief_on_cue() -> None:
    # Current C, on real conversation: a chat turn with a STRICT revision cue ("actually") revises the
    # earlier chat belief, so recall returns only the current value — no LLM at ingest.
    mem = Memory(
        embedder=ConvoUpdateEmbedder(),
        supersede=True,
        supersede_threshold=0.9,
        supersede_conversational=True,
    )
    old = mem.remember("user: my main programming language is python", kind="chat")
    new = mem.remember("user: i actually switched my main language to rust now", kind="chat")

    assert old.superseded_by == new.id
    assert mem.recall("user: my main programming language is python", limit=1)[0].record.id == new.id


def test_conversational_chat_without_cue_does_not_revise() -> None:
    # The guardrail: ordinary chatter (no revision cue) must NOT supersede, or conversational recall
    # collapses (the LoCoMo 0.62 -> 0.00 regression). Same high similarity, but no cue -> inert.
    mem = Memory(
        embedder=ConvoUpdateEmbedder(),
        supersede=True,
        supersede_threshold=0.9,
        supersede_conversational=True,
    )
    old = mem.remember("user: my main programming language is python", kind="chat")
    mem.remember("user: my main language is still python", kind="chat")

    assert old.superseded_by is None


def test_conversational_revision_is_off_by_default() -> None:
    # Default (supersede_conversational=False): chat never revises chat, even with a cue.
    mem = Memory(embedder=ConvoUpdateEmbedder(), supersede=True, supersede_threshold=0.9)
    old = mem.remember("user: my main programming language is python", kind="chat")
    mem.remember("user: i actually switched my main language to rust now", kind="chat")

    assert old.superseded_by is None


class FakeNLI:
    """Stub NLI returning a fixed contradiction score."""

    def __init__(self, contradiction_score: float) -> None:
        self._c = contradiction_score

    def contradiction(self, a: str, b: str) -> float:
        return self._c

    def entailment(self, premise: str, hypothesis: str) -> float:
        return 0.0


def test_conversational_revision_requires_nli_contradiction() -> None:
    # The principled gate: with NLI present, a cue + high similarity is NOT enough — the new turn must
    # actually CONTRADICT the old belief. This is what makes revision safe on diverse conversation.
    mem = Memory(
        embedder=ConvoUpdateEmbedder(),
        supersede=True,
        supersede_threshold=0.9,
        supersede_conversational=True,
        nli=FakeNLI(0.1),  # NLI says: not a contradiction
    )
    old = mem.remember("user: my main programming language is python", kind="chat")
    mem.remember("user: i actually switched my main language to rust now", kind="chat")

    assert old.superseded_by is None


def test_conversational_revision_fires_when_nli_confirms_contradiction() -> None:
    mem = Memory(
        embedder=ConvoUpdateEmbedder(),
        supersede=True,
        supersede_threshold=0.9,
        supersede_conversational=True,
        nli=FakeNLI(0.9),  # NLI confirms contradiction
    )
    old = mem.remember("user: my main programming language is python", kind="chat")
    new = mem.remember("user: i actually switched my main language to rust now", kind="chat")

    assert old.superseded_by == new.id


class FixedReranker:
    """Calibration reranker returning a constant cross-encoder logit for every doc."""

    def __init__(self, score: float) -> None:
        self._score = score

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [self._score] * len(documents)


def test_calibration_floor_adds_firm_abstention_when_score_low() -> None:
    # Calibrated: a low calibration-rerank score means none of the in-context snippets actually answer
    # -> emit a firm caution at the top so the reader abstains instead of confabulating.
    mem = Memory(
        embedder=FixedEmbedder(),
        abstention_relevance_floor=0.5,
        calibration_reranker=FixedReranker(-5.0),  # sigmoid(-5)≈0.007 < floor
    )
    mem.remember("database", kind="fact")

    ctx = mem.assemble("query database", token_budget=100, limit=1)

    assert "no stored memory is relevant" in ctx.lower()
    assert "database" not in ctx  # hard prune: tempting snippet dropped so the reader can't confabulate


def test_calibration_floor_silent_when_score_high() -> None:
    mem = Memory(
        embedder=FixedEmbedder(),
        abstention_relevance_floor=0.5,
        calibration_reranker=FixedReranker(5.0),  # sigmoid(5)≈0.993 > floor
    )
    mem.remember("database", kind="fact")

    ctx = mem.assemble("query database", token_budget=100, limit=1)

    assert "no stored memory is relevant" not in ctx.lower()
    assert "database" in ctx


def test_adapter_reports_current_event_id_after_supersession() -> None:
    adapter = MidasAdapter(
        embedder=FixedEmbedder(),
        limit=1,
        neighbor_window=0,
        rerank=False,
        supersede=True,
    )
    adapter.ingest(
        [
            Event("old-event", "old launch", kind="fact"),
            Event("new-event", "new launch", kind="fact"),
        ]
    )

    result = adapter.query("old launch", token_budget=100)

    assert result.retrieved_event_ids == ["new-event"]
    assert "new launch" in result.context
    assert "old launch" not in result.context
