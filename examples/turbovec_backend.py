"""Compact long-horizon memory with the TurboVec backend.

TurboVec keeps a 2/4-bit-quantized index in RAM (~8-16x smaller than float32) and reranks the top
candidates against full-precision vectors read from disk — so recall matches an exact scan while a
multi-million-turn corpus fits on a laptop (e.g. 10M x 768 float32 = ~31 GB → ~2 GB at 2-bit).

    pip install "midas-memory[local,turbovec]"
    python examples/turbovec_backend.py
"""
from midas import LocalEmbedder, Memory
from midas.turbovec_store import TurboVecStore
from midas.vector_source import SQLiteVectorSource


def main() -> None:
    embedder = LocalEmbedder()  # bge-base, 768-d, ONNX (no torch, no API key)

    # RAM-saving store: the quantized index lives in RAM; full vectors live on disk and are read back
    # only for the candidate set the exact-cosine rerank re-scores. Drop `vector_source` to keep full
    # vectors in RAM instead (simpler, no RAM win). Use bit_width=4 for a quality/size balance.
    store = TurboVecStore(
        dim=embedder.dim,
        bit_width=2,
        rerank_pool=200,
        vector_source=SQLiteVectorSource("vectors.db"),
    )
    mem = Memory(store=store, embedder=embedder)

    mem.remember("The deploy key rotates every 90 days; last rotated 2026-05-01.", kind="fact")
    mem.remember("Vadim prefers TypeScript for the web package.", kind="preference")
    mem.remember("CI runs on the release/* branches only.", kind="note")

    print("recall: 'when does the deploy key rotate?'")
    for hit in mem.recall("when does the deploy key rotate?", limit=3):
        print(f"  {hit.score:.3f}  {hit.record.content}")


if __name__ == "__main__":
    main()
