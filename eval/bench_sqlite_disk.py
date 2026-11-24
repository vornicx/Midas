"""Measure exact SQLite search against a reproducible, synthetic memory corpus."""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import struct
import tempfile
import threading
import time
from pathlib import Path

from midas.embeddings import l2_normalize
from midas.sqlite_store import SQLiteStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=100_000)
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--queries", type=int, default=12)
    parser.add_argument("--scoped", action="store_true", help="filter to one of two projects")
    args = parser.parse_args()
    if min(args.records, args.dimensions, args.queries) < 1:
        parser.error("records, dimensions, and queries must be positive")
    try:
        import psutil
    except ImportError:
        parser.error("Install psutil to measure process RAM (python -m pip install psutil).")

    randomizer = random.Random(7)
    vectors = [l2_normalize([randomizer.uniform(-1, 1) for _ in range(args.dimensions)])
               for _ in range(256)]
    blobs = [struct.pack(f"<{args.dimensions}f", *vector) for vector in vectors]
    process = psutil.Process()
    with tempfile.TemporaryDirectory(prefix="midas-sqlite-bench-") as directory:
        db = Path(directory) / "memory.sqlite3"
        SQLiteStore(db).close()
        with sqlite3.connect(db) as conn:
            conn.executemany(
                """INSERT INTO memories
                   (id, content, kind, importance, created_at, updated_at, embedding, metadata_json)
                   VALUES (?, ?, 'fact', 3, 1, 1, ?, ?)""",
                ((str(i), f"Decision {i}: service {i % 41} deploys through staging.",
                  blobs[i % len(blobs)], '{"namespace":"one"}' if i % 2 else '{"namespace":"two"}')
                 for i in range(args.records)),
            )
        conn.close()

        mib = 1024**2
        initial = process.memory_info().rss
        peak = initial
        finished = threading.Event()

        def sample() -> None:
            nonlocal peak
            while not finished.wait(0.005):
                peak = max(peak, process.memory_info().rss)

        sampler = threading.Thread(target=sample)
        sampler.start()
        try:
            start = time.perf_counter()
            store = SQLiteStore(db)
            startup_ms = (time.perf_counter() - start) * 1000
            opened = process.memory_info().rss
            elapsed = []
            predicate = (lambda r: r.metadata.get("namespace") == "one") if args.scoped else None
            cpu_begin = time.process_time()
            for i in range(args.queries):
                start = time.perf_counter()
                vector = vectors[(i * 13 + 1) % len(vectors)]
                search_namespace = getattr(store, "search_namespace", None)
                if args.scoped and search_namespace is not None:
                    hits = search_namespace(vector, limit=10, namespace="one")
                else:
                    hits = store.search(vector, limit=10, predicate=predicate)
                elapsed.append((time.perf_counter() - start) * 1000)
                if not hits:
                    raise RuntimeError("No search results")
            cpu_ms = (time.process_time() - cpu_begin) * 1000 / args.queries
            store.close()
        finally:
            finished.set()
            sampler.join()

        elapsed.sort()
        percentile = max(0, (95 * len(elapsed) + 99) // 100 - 1)
        print(json.dumps({
            "records": args.records, "dimensions": args.dimensions,
            "queries": args.queries, "scoped": args.scoped,
            "open_ms": round(startup_ms, 2), "p95_search_ms": round(elapsed[percentile], 2),
            "mean_cpu_ms_per_query": round(cpu_ms, 2),
            "rss_before_mib": round(initial / mib, 2), "rss_after_open_mib": round(opened / mib, 2),
            "peak_rss_mib": round(peak / mib, 2),
        }, indent=2))


if __name__ == "__main__":
    main()
