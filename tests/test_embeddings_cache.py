from __future__ import annotations

import os

from midas import DiskCachedEmbedder, configure_local_model_cache


class CountingEmbedder:
    dim = 3

    def __init__(self) -> None:
        self.batch_calls = 0
        self.single_calls = 0
        self.texts: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.single_calls += 1
        self.texts.append(text)
        return [float(len(text)), 1.0, 0.0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls += 1
        self.texts.extend(texts)
        return [[float(len(text)), 1.0, 0.0] for text in texts]


def test_configure_local_model_cache_uses_project_env_overrides(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "hf-cache"
    tmp_root = tmp_path / "tmp"
    monkeypatch.setenv("MIDAS_CACHE_ROOT", str(cache_root))
    monkeypatch.setenv("MIDAS_TMP_DIR", str(tmp_root))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("TMP", raising=False)
    monkeypatch.delenv("TEMP", raising=False)

    cache_dir = configure_local_model_cache()

    assert cache_dir == str(cache_root / "fastembed")
    assert (cache_root / "fastembed").exists()
    assert tmp_root.exists()
    assert os.environ["HF_HOME"] == str(cache_root)
    assert os.environ["HF_HUB_CACHE"] == str(cache_root / "hub")
    assert os.environ["TMP"] == str(tmp_root)
    assert os.environ["TEMP"] == str(tmp_root)


def test_configure_local_model_cache_preserves_explicit_hf_env(tmp_path, monkeypatch) -> None:
    explicit_hf = tmp_path / "explicit-hf"
    explicit_tmp = tmp_path / "explicit-tmp"
    monkeypatch.setenv("MIDAS_CACHE_ROOT", str(tmp_path / "midas-cache"))
    monkeypatch.setenv("MIDAS_TMP_DIR", str(tmp_path / "midas-tmp"))
    monkeypatch.setenv("HF_HOME", str(explicit_hf))
    monkeypatch.setenv("TMP", str(explicit_tmp))

    configure_local_model_cache(tmp_path / "fastembed-models")

    assert os.environ["HF_HOME"] == str(explicit_hf)
    assert os.environ["TMP"] == str(explicit_tmp)


def test_disk_cached_embedder_reuses_persisted_vectors(tmp_path) -> None:
    cache_path = tmp_path / "embeddings.sqlite3"
    inner = CountingEmbedder()
    cached = DiskCachedEmbedder(inner, path=cache_path, namespace="test")

    vectors = cached.embed_many(["alpha", "beta", "alpha"])

    assert vectors == [[5.0, 1.0, 0.0], [4.0, 1.0, 0.0], [5.0, 1.0, 0.0]]
    assert inner.batch_calls == 1
    assert inner.single_calls == 0
    assert inner.texts == ["alpha", "beta"]

    second_inner = CountingEmbedder()
    second_cached = DiskCachedEmbedder(second_inner, path=cache_path, namespace="test")

    assert second_cached.embed_many(["beta", "alpha"]) == [[4.0, 1.0, 0.0], [5.0, 1.0, 0.0]]
    assert second_inner.batch_calls == 0
    assert second_inner.single_calls == 0


def test_cache_refuses_mislabelled_dimension(tmp_path):
    """An embedder lying about `dim` must fail at write time, not poison the disk cache."""
    import pytest

    from midas.embeddings import DiskCachedEmbedder

    class LyingEmbedder:
        dim = 768  # claims 768...

        def embed(self, text):
            return [0.1] * 384  # ...but produces 384

        def embed_many(self, texts):
            return [self.embed(t) for t in texts]

    cached = DiskCachedEmbedder(LyingEmbedder(), path=tmp_path / "cache.sqlite3")
    with pytest.raises(ValueError, match="dim"):
        cached.embed("hello")
