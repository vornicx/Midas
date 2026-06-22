"""Pluggable embeddings.

`HashingEmbedder` is a dependency-free, deterministic, offline stand-in so the
whole harness runs anywhere with zero setup. It is essentially a vectorized
bag-of-words — good enough to exercise the pipeline, NOT a real semantic model.
Swap in `OpenAIEmbedder` (or a local BGE/e5) for genuine semantic retrieval;
that swap is the entire reason embeddings live behind a Protocol.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Protocol, runtime_checkable

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens of length > 2 (Unicode-aware)."""
    return [w for w in _WORD.findall(text.lower()) if len(w) > 2]


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Offline, deterministic bag-of-words hashed into a fixed-dim unit vector."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            h = int.from_bytes(digest[:8], "big")
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        return l2_normalize(vec)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class OpenAIEmbedder:
    """Real semantic embeddings via OpenAI. Requires `OPENAI_API_KEY` and the
    `openai` package (`uv pip install openai`)."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI  # lazy import — optional dependency

        self._client = OpenAI()
        self.model = model
        self.dim = 1536

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self.model, input=text)
        return l2_normalize(list(resp.data[0].embedding))

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [l2_normalize(list(item.embedding)) for item in resp.data]


class TruncatedEmbedder:
    """Matryoshka dimensionality truncation: keep the first `dim` components of an inner embedder's
    vectors and renormalize. MRL-trained models (jina-v3, mxbai, nomic) retain most retrieval
    quality at a fraction of the dimensions → faster cosine and less RAM/disk; non-MRL models
    degrade (the eval measures which). Implements the Embedder protocol, so it caches and slots in
    like any other embedder — `model_name`/`dim` differ from the inner model, giving it its own
    cache namespace."""

    def __init__(self, inner: object, dim: int) -> None:
        self.inner = inner
        self.dim = dim
        self.model_name = f"{getattr(inner, 'model_name', type(inner).__name__)}@mrl{dim}"
        self.max_text_chars = getattr(inner, "max_text_chars", None)
        self.batch_size = getattr(inner, "batch_size", None)

    def _truncate(self, vec: list[float]) -> list[float]:
        return l2_normalize(list(vec[: self.dim]))

    def embed(self, text: str) -> list[float]:
        return self._truncate(self.inner.embed(text))

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self._truncate(v) for v in self.inner.embed_many(texts)]


class Model2VecEmbedder:
    """Static (distilled) embeddings via Model2Vec — no transformer forward pass at encode time, so
    it is ~CPU-instant. A speed/footprint tier: lower quality than a bi-encoder but orders of
    magnitude faster, useful for cheap candidate generation. Needs `pip install model2vec`."""

    def __init__(self, model_name: str = "minishlab/potion-base-8M") -> None:
        from model2vec import StaticModel  # lazy import — optional dependency

        self._model = StaticModel.from_pretrained(model_name)
        self.model_name = model_name
        probe = self._model.encode(["x"])
        self.dim = int(getattr(self._model, "dim", 0)) or len(probe[0])

    def embed(self, text: str) -> list[float]:
        return l2_normalize([float(v) for v in self._model.encode([text])[0]])

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [l2_normalize([float(v) for v in vec]) for vec in self._model.encode(texts)]


def _default_embedding_cache_path() -> Path:
    return _default_cache_root() / "midas-embeddings.sqlite3"


def _cache_namespace(embedder: object) -> str:
    parts = [
        f"{type(embedder).__module__}.{type(embedder).__qualname__}",
        f"dim={getattr(embedder, 'dim', 'unknown')}",
    ]
    for attr in ("model_name", "max_text_chars"):
        value = getattr(embedder, attr, None)
        if value is not None:
            parts.append(f"{attr}={value}")
    return "|".join(parts)


def _encode_vector(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *(float(v) for v in vec))


def _decode_vector(blob: bytes, *, dim: int) -> list[float]:
    if len(blob) != dim * 4:
        raise ValueError("cached embedding has an unexpected dimension")
    return [float(v) for v in struct.unpack(f"<{dim}f", blob)]


class DiskCachedEmbedder:
    """Persistent SQLite cache wrapper for expensive embedders.

    The key includes the embedder namespace and exact input text hash, so changing the
    local model or truncation limit gets a separate cache namespace by default.
    """

    def __init__(
        self,
        inner: Embedder,
        *,
        path: str | Path | None = None,
        namespace: str | None = None,
    ) -> None:
        self.inner = inner
        self.dim = inner.dim
        self.namespace = namespace or _cache_namespace(inner)
        self.path = Path(path) if path is not None else _default_embedding_cache_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    namespace TEXT NOT NULL,
                    text_sha256 TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at INTEGER NOT NULL DEFAULT (unixepoch()),
                    PRIMARY KEY (namespace, text_sha256)
                )
                """
            )
            self._conn.commit()
        self.hits = 0
        self.misses = 0

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        keys = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
        cached = self._read_many(set(keys))
        missing: dict[str, str] = {}
        for key, text in zip(keys, texts):
            if key not in cached and key not in missing:
                missing[key] = text

        self.hits += len(texts) - sum(1 for key in keys if key not in cached)
        self.misses += len(missing)

        if missing:
            missing_keys = list(missing)
            missing_texts = [missing[key] for key in missing_keys]
            embed_many = getattr(self.inner, "embed_many", None)
            vectors = (
                embed_many(missing_texts)
                if callable(embed_many)
                else [self.inner.embed(text) for text in missing_texts]
            )
            if len(vectors) != len(missing_keys):
                raise ValueError("cached embedder inner returned the wrong number of embeddings")
            self._write_many(zip(missing_keys, vectors))
            cached.update(zip(missing_keys, vectors))

        return [cached[key] for key in keys]

    def _read_many(self, keys: set[str]) -> dict[str, list[float]]:
        if not keys:
            return {}
        rows: dict[str, list[float]] = {}
        ordered = list(keys)
        with self._lock:
            for i in range(0, len(ordered), 500):
                chunk = ordered[i : i + 500]
                placeholders = ",".join("?" for _ in chunk)
                params = [self.namespace, *chunk]
                cur = self._conn.execute(
                    f"""
                    SELECT text_sha256, dim, embedding
                    FROM embeddings
                    WHERE namespace = ? AND text_sha256 IN ({placeholders})
                    """,
                    params,
                )
                for key, dim, blob in cur.fetchall():
                    if dim == self.dim:
                        rows[key] = _decode_vector(blob, dim=self.dim)
        return rows

    def _write_many(self, items) -> None:
        items = list(items)
        for _, vec in items:
            if len(vec) != self.dim:
                # Fail loudly at write time: an embedder lying about its `dim` (the old
                # LocalEmbedder hardcoded 768 for every model) would otherwise poison the
                # on-disk cache and only blow up sessions later, at read time.
                raise ValueError(
                    f"embedder produced a {len(vec)}-d vector but declares dim={self.dim}; "
                    "refusing to write a mislabelled cache row"
                )
        rows = [
            (self.namespace, key, self.dim, _encode_vector(vec))
            for key, vec in items
        ]
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings(namespace, text_sha256, dim, embedding)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()


def _default_cache_root() -> Path:
    if root := os.getenv("MIDAS_CACHE_ROOT"):
        return Path(root)
    if os.name == "nt" and Path("D:/").exists():
        return Path("D:/hf-cache")
    return Path.home() / ".cache" / "midas"


def _default_tmp_dir() -> Path:
    if tmp := os.getenv("MIDAS_TMP_DIR"):
        return Path(tmp)
    if os.name == "nt" and Path("D:/").exists():
        return Path("D:/tmp")
    return _default_cache_root() / "tmp"


def configure_local_model_cache(cache_dir: str | Path | None = None) -> str:
    """Return a fastembed cache dir and set safe local cache env defaults."""
    cache_root = _default_cache_root()
    root = Path(cache_dir) if cache_dir is not None else cache_root / "fastembed"
    tmp = _default_tmp_dir()
    root.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("TMP", str(tmp))
    os.environ.setdefault("TEMP", str(tmp))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return str(root)


class LocalEmbedder:
    """Real local semantic embeddings via fastembed (ONNX — no API key, no torch).
    First use downloads a small model (~130 MB). Install: `uv pip install fastembed`."""

    _MAX_TEXT_CHARS = 2000

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        *,
        cache_dir: str | Path | None = None,
        batch_size: int = 32,
        max_text_chars: int | None = None,
    ) -> None:
        self.cache_dir = configure_local_model_cache(cache_dir)
        from fastembed import TextEmbedding  # lazy import — optional dependency

        self._model = TextEmbedding(model_name=model_name, cache_dir=self.cache_dir)
        self.model_name = model_name
        # Dimension from the model's own registry entry — hardcoding 768 poisoned the disk cache
        # for any non-default model (e.g. multilingual MiniLM is 384-d: blobs were written with a
        # wrong dim label and failed to decode on the next session).
        self.dim = next(
            (
                int(m["dim"])
                for m in TextEmbedding.list_supported_models()
                if m["model"] == model_name
            ),
            0,
        ) or len(next(iter(self._model.embed(["probe"], batch_size=1))))
        self.batch_size = batch_size
        self.max_text_chars = max_text_chars or self._MAX_TEXT_CHARS

    def embed(self, text: str) -> list[float]:
        vec = next(iter(self._model.embed([text[: self.max_text_chars]], batch_size=1)))
        return l2_normalize([float(x) for x in vec])

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        capped = [text[: self.max_text_chars] for text in texts]
        return [
            l2_normalize([float(x) for x in vec])
            for vec in self._model.embed(capped, batch_size=self.batch_size)
        ]


class LocalReranker:
    """Local cross-encoder reranker (fastembed/ONNX, no API key, no torch). Reorders
    candidate documents by relevance to the query — sharper precision than bi-encoder
    cosine. First use downloads a small model. Used by Midas to refine the recall pool."""

    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.cache_dir = configure_local_model_cache(cache_dir)
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # lazy, optional

        self._model = TextCrossEncoder(model_name=model_name, cache_dir=self.cache_dir)
        self.model_name = model_name

    # Cross-encoders cap the query+doc PAIR at ~512 tokens; oversized input crashes the ONNX
    # runtime (seen on long real turns). Cap defensively — affects only the rerank score, not
    # the stored/assembled content.
    _MAX_QUERY_CHARS = 400
    _MAX_DOC_CHARS = 1200

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Return a relevance score per document (higher = more relevant), in order.
        Length-capped to stay within the cross-encoder context; degrades to neutral scores on
        any runtime error rather than crashing the whole query."""
        if not documents:
            return []
        q = query[: self._MAX_QUERY_CHARS]
        docs = [d[: self._MAX_DOC_CHARS] for d in documents]
        try:
            return [float(s) for s in self._model.rerank(q, docs)]
        except Exception:
            return [0.0] * len(docs)


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Inputs assumed L2-normalized; result clamped to [-1, 1]."""
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
