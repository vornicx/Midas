"""Pinned standing-directive channel: cue detection at ingest + reserved context slots."""
from midas import HashingEmbedder, Memory
from midas.importance import is_standing_instruction


def test_standing_instruction_detector():
    assert is_standing_instruction("From now on, reply in Spanish.")
    assert is_standing_instruction("Always format code with type hints.")
    assert is_standing_instruction("Make sure to include error handling in every snippet.")
    assert not is_standing_instruction("Should I always use type hints?")  # question
    assert not is_standing_instruction("I never went to Paris.")  # narrative, not a directive
    assert not is_standing_instruction("haha ok cool")


def test_ingest_tags_standing_directives():
    mem = Memory(embedder=HashingEmbedder(), detect_standing=True)
    rec = mem.remember("From now on, always format code with type hints.", kind="chat")
    assert rec.metadata.get("standing") is True
    plain = mem.remember("We talked about the weather.", kind="chat")
    assert "standing" not in plain.metadata

    many = mem.remember_many([{"content": "Remember to use metric units in every answer."}])
    assert many[0].metadata.get("standing") is True


def test_pinned_directives_enter_context_regardless_of_relevance():
    mem = Memory(embedder=HashingEmbedder(), detect_standing=True)
    mem.remember("From now on, always reply in Spanish.", kind="chat", importance=4)
    for i in range(30):
        mem.remember(f"database migration note number {i} about postgres tables", kind="note")

    block = mem.build_context("how do I migrate the postgres schema?", token_budget=300, pinned_limit=2)
    assert any("reply in Spanish" in r.content for r in block.records), (
        "the standing directive must be pinned in even though the query is unrelated"
    )
    # And without pinning, vector relevance alone misses it on this unrelated query.
    plain = mem.build_context("how do I migrate the postgres schema?", token_budget=300)
    assert not any("reply in Spanish" in r.content for r in plain.records)


def test_pinned_respects_namespace_filter():
    mem = Memory(embedder=HashingEmbedder(), detect_standing=True)
    mem.remember("Always use tabs for indentation.", kind="chat", metadata={"namespace": "a"})
    block = mem.build_context(
        "anything", token_budget=200, pinned_limit=2, metadata_filter={"namespace": "b"}
    )
    assert not any("tabs" in r.content for r in block.records)
