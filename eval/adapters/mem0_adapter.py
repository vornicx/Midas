"""Mem0 competitor adapter.

Mem0 ingests dialogue and uses an LLM to extract/update structured 'memories',
then retrieves those facts at query time. Configured here for a *fair* comparison
to Atlas: the SAME embedder (fastembed bge-small) + a local Qdrant store, with Groq
as the extraction LLM. recall@k is N/A (Mem0 returns synthesized facts, not source
turn ids) — the cross-system metric is end-to-end judged correctness.

Needs: `uv pip install mem0ai groq` and a GROQ_API_KEY (loaded from .env).
"""
from __future__ import annotations

import os
import tempfile

from ..schema import Event
from .base import RetrievalResult

_EMBED_MODEL = "BAAI/bge-base-en-v1.5"  # matched to Midas for a fair embedder comparison
_EMBED_DIMS = 768


class Mem0Adapter:
    name = "mem0"
    tracks_event_ids = False  # synthesizes facts, not turn ids -> recall@k N/A
    uses_internal_llm = True  # LLM extraction at ingest - the cost we measure against

    def __init__(self, *, model: str | None = None, search_limit: int = 20) -> None:
        os.environ.setdefault("MEM0_TELEMETRY", "False")
        # LLM provider for Mem0's per-session extraction. Default 'openai' auto-routes to
        # OpenRouter when OPENROUTER_API_KEY is set. Set MEM0_LLM_PROVIDER=ollama for an
        # unlimited LOCAL llm — the robust path, since free cloud tiers (Groq 6k TPM,
        # Gemini quota, OpenRouter free) rate-limit the heavy 8k-token extraction calls.
        # Auto-detect: OpenCode (Kimi) → Ollama fallback
        if os.getenv("OPENCODEGO_API_KEY"):
            self._provider = "opencode"
            default_model = "kimi-k2.6"
        elif os.getenv("MEM0_LLM_PROVIDER"):
            self._provider = os.getenv("MEM0_LLM_PROVIDER")
            default_model = "llama3.1:8b" if self._provider == "ollama" else "meta-llama/llama-3.3-70b-instruct:free"
        else:
            self._provider = "ollama"
            default_model = "llama3.1:8b"
        self._model = model or os.getenv("MEM0_MODEL") or default_model
        self._search_limit = search_limit
        self._uid = "bench"
        self.reset()

    def _llm_config(self) -> dict:
        if self._provider == "opencode":
            # Mem0's 'openai' provider auto-detects OPENROUTER_API_KEY and routes there.
            # We must explicitly clear it to force our custom base_url.
            os.environ.pop("OPENROUTER_API_KEY", None)
            return {
                "provider": "openai",
                "config": {
                    "model": self._model,
                    "temperature": 0.0,
                    "api_key": os.environ["OPENCODEGO_API_KEY"],
                    "openai_base_url": "https://opencode.ai/zen/go/v1",
                },
            }
        if self._provider == "ollama":
            return {
                "provider": "ollama",
                "config": {
                    "model": self._model,
                    "temperature": 0.0,
                    "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                },
            }
        # Fallback: 'openai' provider (auto-routes to OpenRouter if OPENROUTER_API_KEY set)
        return {"provider": "openai", "config": {"model": self._model, "temperature": 0.0}}

    def _config(self, path: str) -> dict:
        return {
            "llm": self._llm_config(),
            "embedder": {"provider": "fastembed", "config": {"model": _EMBED_MODEL}},
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "bench",
                    "embedding_model_dims": _EMBED_DIMS,
                    "path": path,
                },
            },
        }

    def reset(self) -> None:
        from mem0 import Memory

        # Fresh local Qdrant path per reset (Qdrant local mode locks a path to one client).
        path = tempfile.mkdtemp(prefix="mem0_qdrant_")
        self._mem = Memory.from_config(self._config(path))

    def ingest(self, events: list[Event]) -> None:
        # Group by session and add per-session to bound the number of LLM calls.
        batches: dict[str, list[Event]] = {}
        order: list[str] = []
        for event in events:
            key = str(event.metadata.get("session") or event.metadata.get("session_id") or "default")
            if key not in batches:
                batches[key] = []
                order.append(key)
            batches[key].append(event)

        for key in order:
            messages = [{"role": "user", "content": e.content} for e in batches[key]]
            try:
                self._mem.add(messages, user_id=self._uid)
            except Exception as exc:  # one bad batch shouldn't abort the whole ingest
                print(f"[mem0] add() failed for {key}: {exc}")

    def query(self, question: str, *, token_budget: int, now: float | None = None) -> RetrievalResult:
        from midas import approx_tokens

        res = self._mem.search(question, filters={"user_id": self._uid}, limit=self._search_limit)
        items = res.get("results", []) if isinstance(res, dict) else (res or [])

        lines: list[str] = []
        used = 0
        for item in items:
            text = item.get("memory") or item.get("text") or "" if isinstance(item, dict) else str(item)
            if not text:
                continue
            line = f"- {text}"
            cost = approx_tokens(line)
            if used + cost > token_budget:
                break
            lines.append(line)
            used += cost
        context = ("# Memory (Mem0)\n" + "\n".join(lines)) if lines else ""
        return RetrievalResult(context=context, retrieved_event_ids=[], tokens=used)


if __name__ == "__main__":
    from ..llm import load_dotenv

    load_dotenv()
    adapter = Mem0Adapter()
    adapter.ingest(
        [
            Event("t1", "Alex: My favorite language is Python.", speaker="Alex", metadata={"session": "session_1"}),
            Event("t2", "Sam: Got it. By the way the launch is on March 3.", speaker="Sam", metadata={"session": "session_1"}),
        ]
    )
    result = adapter.query("What is Alex's favorite programming language?", token_budget=200)
    print("context:\n" + (result.context or "(empty)"))
    print("tokens:", result.tokens)
