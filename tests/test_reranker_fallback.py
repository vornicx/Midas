import pytest

from midas import HashingEmbedder, LocalReranker, Memory, embeddings


def test_optional_reranker_load_failure_keeps_semantic_embedder(monkeypatch):
    semantic_embedder = object()

    def unavailable(**kwargs):
        raise RuntimeError("cross-encoder unavailable")

    monkeypatch.setenv("MIDAS_MCP_EMBEDDER", "local")
    monkeypatch.setattr(embeddings, "LocalEmbedder", lambda **kwargs: semantic_embedder)
    monkeypatch.setattr(embeddings, "LocalReranker", unavailable)
    assert embeddings.mcp_embeddings() == (semantic_embedder, None)
    monkeypatch.setattr(embeddings, "LocalEmbedder", unavailable)
    model, reranker = embeddings.mcp_embeddings()
    assert isinstance(model, HashingEmbedder) and reranker is None


@pytest.mark.parametrize("output", [RuntimeError("ONNX runtime failure"), [0.0], [float("nan")] * 2])
def test_reranker_runtime_failure_preserves_recall_and_context(output):
    class Model:
        def rerank(self, query, documents):
            if isinstance(output, Exception):
                raise output
            return output

    mem = Memory(embedder=HashingEmbedder())
    mem.remember("PostgreSQL is the production database.", kind="fact", importance=4)
    mem.remember("Lunch reservations close at noon.", kind="note")
    query = "PostgreSQL database"
    recall = mem.recall(query, now=1000)
    context = mem.build_context(query, now=1000)
    assert len(recall) == 1

    local = LocalReranker.__new__(LocalReranker)
    local._model = Model()
    mem.reranker = local
    assert mem.recall(query, now=1000) == recall
    assert mem.build_context(query, now=1000).text == context.text


def test_shared_engine_keeps_legacy_hashing_vectors_usable(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from midas import mcp_server
    from midas.sqlite_store import SQLiteStore

    path = tmp_path / "legacy.sqlite3"
    store = SQLiteStore(path)
    before = Memory(store=store, embedder=HashingEmbedder())
    record = before.remember("The EDI transport uses AS2.", kind="fact")
    store.close()

    def unexpected_model():
        pytest.fail("An existing hashing store should use its original embedder.")

    monkeypatch.setenv("MIDAS_MCP_DB", str(path))
    monkeypatch.setenv("MIDAS_MCP_EMBEDDER", "local")
    monkeypatch.setattr(mcp_server, "mcp_embeddings", unexpected_model)
    after = mcp_server.build_memory()
    assert after.recall("EDI transport")[0].record.id == record.id
    assert after.store.embedding_dimensions() == {256}
    after.store.close()
