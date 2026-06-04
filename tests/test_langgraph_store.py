"""MidasStore as a LangGraph BaseStore — put / get / semantic search / delete / namespaces."""
import pytest

pytest.importorskip("langgraph")

from midas.integrations.langgraph_store import MidasStore  # noqa: E402


def test_put_get_and_semantic_search():
    store = MidasStore()  # offline hashing embedder
    store.put(("user", "1"), "db", {"text": "the primary database is postgresql"})
    store.put(("user", "1"), "lunch", {"text": "we chatted about lunch options"})

    got = store.get(("user", "1"), "db")
    assert got is not None and "postgresql" in got.value["text"]

    hits = store.search(("user",), query="which database did we choose?", limit=1)
    assert hits and "postgresql" in hits[0].value["text"]
    assert hits[0].score is not None  # semantic search yields a score


def test_delete_and_list_namespaces():
    store = MidasStore()
    store.put(("a", "x"), "k1", {"text": "alpha"})
    store.put(("b",), "k2", {"text": "beta"})
    store.put(("a", "x"), "k1", None)  # value=None deletes

    assert store.get(("a", "x"), "k1") is None
    namespaces = store.list_namespaces()
    assert ("b",) in namespaces
    assert ("a", "x") not in namespaces  # its only item was deleted
