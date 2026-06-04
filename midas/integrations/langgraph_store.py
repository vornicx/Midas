"""Use Midas as a LangGraph long-term-memory store.

LangGraph agents read/write long-term memory through a `BaseStore`. `MidasStore` implements that
interface backed by Midas, so recall is **semantic, local, and provenance-bearing — with no LLM at
ingest or query**. Each LangGraph item (namespace, key, value) maps to one Midas record; the
searchable text is the value (or its `index` fields).

    from midas.integrations.langgraph_store import MidasStore
    from midas import LocalEmbedder, LocalReranker, Memory

    store = MidasStore()  # offline hashing by default; pass a configured Memory for semantic quality
    store.put(("user", "123"), "pref", {"text": "prefers dark mode and concise answers"})
    hits = store.search(("user", "123"), query="ui preferences", limit=3)

Install: `uv pip install -e ".[langgraph,local]"`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    SearchItem,
    SearchOp,
)

from ..embeddings import HashingEmbedder
from ..memory import Memory


def _dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _searchable_text(value: dict[str, Any], index: Any) -> str:
    """Text used for semantic recall: the `index` fields if given, else all values."""
    if index:
        parts = [str(value.get(field, "")) for field in index]
    else:
        parts = [str(v) for v in value.values()]
    return " ".join(p for p in parts if p)


class MidasStore(BaseStore):
    """A LangGraph `BaseStore` backed by Midas (semantic, local, no-LLM recall).

    Note: handles Put/Get/Search/ListNamespaces. TTL and `ListNamespacesOp.match_conditions` are not
    yet implemented (max_depth/limit/offset are); semantic search needs an embedder on the `Memory`.
    """

    def __init__(self, memory: Memory | None = None) -> None:
        self._mem = memory or Memory(embedder=HashingEmbedder())
        self._ids: dict[tuple[tuple[str, ...], str], str] = {}  # (namespace, key) -> record id

    # The one abstract method (concrete get/put/search/list_namespaces delegate here).
    def batch(self, ops: Iterable[Op]) -> list[Any]:
        results: list[Any] = []
        for op in ops:
            if isinstance(op, PutOp):
                self._put(op)
                results.append(None)
            elif isinstance(op, GetOp):
                results.append(self._get(op))
            elif isinstance(op, SearchOp):
                results.append(self._search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._list_namespaces(op))
            else:  # pragma: no cover - unknown op
                results.append(None)
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Any]:
        return self.batch(ops)  # Midas is synchronous + fast; no benefit to a thread here

    # -- op handlers ---------------------------------------------------------------------------
    def _put(self, op: PutOp) -> None:
        nk = (op.namespace, op.key)
        old = self._ids.pop(nk, None)
        if old is not None:
            self._mem.store.delete(old)
        if op.value is None:  # PutOp with value=None means delete
            return
        record = self._mem.remember(
            _searchable_text(op.value, op.index) or op.key,
            kind="note",
            metadata={"ns": list(op.namespace), "key": op.key, "value": op.value},
        )
        self._ids[nk] = record.id

    def _get(self, op: GetOp) -> Item | None:
        rid = self._ids.get((op.namespace, op.key))
        rec = self._mem.store.get(rid) if rid is not None else None
        return self._to_item(rec) if rec is not None else None

    def _search(self, op: SearchOp) -> list[SearchItem]:
        prefix = op.namespace_prefix
        items: list[SearchItem] = []
        if op.query:
            hits = self._mem.recall(op.query, limit=op.limit + op.offset + 20)
            for hit in hits:
                if self._in_prefix(hit.record, prefix) and self._match_filter(hit.record, op.filter):
                    items.append(self._to_search_item(hit.record, float(hit.score)))
        else:
            for rec in self._mem.store.all():
                if self._in_prefix(rec, prefix) and self._match_filter(rec, op.filter):
                    items.append(self._to_search_item(rec, None))
            items.sort(key=lambda si: si.created_at, reverse=True)
        return items[op.offset : op.offset + op.limit]

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        seen: set[tuple[str, ...]] = set()
        for rec in self._mem.store.all():
            ns = tuple(rec.metadata.get("ns", []))
            if op.max_depth is not None:
                ns = ns[: op.max_depth]
            seen.add(ns)
        return sorted(seen)[op.offset : op.offset + op.limit]

    # -- helpers -------------------------------------------------------------------------------
    @staticmethod
    def _in_prefix(rec, prefix: tuple[str, ...]) -> bool:
        ns = tuple(rec.metadata.get("ns", []))
        return ns[: len(prefix)] == prefix

    @staticmethod
    def _match_filter(rec, filter: dict[str, Any] | None) -> bool:
        if not filter:
            return True
        value = rec.metadata.get("value", {})
        return all(value.get(k) == v for k, v in filter.items())

    def _to_item(self, rec) -> Item:
        md = rec.metadata
        return Item(
            value=md.get("value", {}), key=md.get("key", ""), namespace=tuple(md.get("ns", [])),
            created_at=_dt(rec.created_at), updated_at=_dt(rec.updated_at),
        )

    def _to_search_item(self, rec, score: float | None) -> SearchItem:
        md = rec.metadata
        return SearchItem(
            namespace=tuple(md.get("ns", [])), key=md.get("key", ""), value=md.get("value", {}),
            created_at=_dt(rec.created_at), updated_at=_dt(rec.updated_at), score=score,
        )
