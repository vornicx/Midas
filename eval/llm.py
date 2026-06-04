"""Minimal OpenAI-compatible chat client (httpx) + .env loader.

Used for the LLM-judge correctness metric and (later) any LLM-backed adapter.
Picks a provider from env: GROQ > OPENROUTER > MISTRAL. Override with JUDGE_MODEL.
No SDK dependency — httpx ships as a fastembed dep.

    uv run --no-sync python -m eval.llm     # smoke-test the configured provider
"""
from __future__ import annotations

import os
import time
from pathlib import Path

# (env var, base url, default model) — tried in order.
_PROVIDERS = [
    ("OPENCODEGO_API_KEY", "https://opencode.ai/zen/go/v1", "kimi-k2.6"),
    ("GROQ_API_KEY", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct"),
    ("MISTRAL_API_KEY", "https://api.mistral.ai/v1", "mistral-large-latest"),
]


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    if not value:
        return None
    try:
        return min(float(value), 30.0)
    except ValueError:
        return None


def load_dotenv(path: str | Path | None = None) -> None:
    path = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


class ChatLLM:
    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        import httpx  # optional dependency; only needed when --judge is used

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._throttle = 0.4  # seconds between successful calls (smooth RPM)
        self._seed: int | None = None  # set for reproducible local inference (Ollama)
        self.max_workers = 6  # judge concurrency; set 1 for local models (batching breaks determinism)
        self._client = httpx.Client(timeout=120.0)

    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 512
    ) -> str:
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._seed is not None:
            payload["seed"] = self._seed
        import httpx  # always present (fastembed dep); needed here for the retry exception types

        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(7):
            try:
                resp = self._client.post(
                    f"{self.base_url}/chat/completions", headers=headers, json=payload
                )
            except httpx.TransportError:  # read timeout / connection reset — back off and retry
                time.sleep(min(3.0 * (attempt + 1), 20.0))
                continue
            if resp.status_code == 429:  # rate limited — wait (honor Retry-After) and retry
                time.sleep(_retry_after(resp) or min(3.0 * (attempt + 1), 20.0))
                continue
            if resp.status_code >= 500:  # transient server error (502/503/529) — back off and retry
                time.sleep(min(3.0 * (attempt + 1), 20.0))
                continue
            resp.raise_for_status()
            if self._throttle:
                time.sleep(self._throttle)
            msg = resp.json()["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            # Reasoning models sometimes spend the whole token budget in a separate reasoning channel
            # and return empty content; fall back to that channel so we still have an answer to grade.
            if not content:
                content = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
            return content
        print("[llm] gave up after repeated timeouts/429s — scoring this call as empty")
        return ""


def from_env(model: str | None = None) -> ChatLLM:
    load_dotenv()
    override_model = model or os.environ.get("JUDGE_MODEL")
    # JUDGE_PROVIDER pins a specific provider (e.g. "groq") regardless of order — used to run
    # correctness against a STRONG + reasonably-fixed hosted reader (Groq Llama-3.3-70B at temp 0)
    # instead of the default drifting MoE gateway. Matches on the env-var name substring.
    preferred = os.environ.get("JUDGE_PROVIDER")
    providers = _PROVIDERS
    if preferred:
        providers = sorted(_PROVIDERS, key=lambda p: 0 if preferred.lower() in p[0].lower() else 1)
    for env_key, base_url, default_model in providers:
        key = os.environ.get(env_key)
        if key:
            return ChatLLM(base_url=base_url, api_key=key, model=override_model or default_model)
    raise RuntimeError(
        "No LLM key found. Set GROQ_API_KEY / OPENROUTER_API_KEY / MISTRAL_API_KEY in .env."
    )


def from_ollama(
    model: str = "sam860/dolphin3-qwen2.5:1.5b", *, base_url: str = "http://localhost:11434/v1"
) -> ChatLLM:
    """Local Ollama reader/judge — REPRODUCIBLE (temp 0 + pinned seed, no API/MoE drift).
    Weaker than a frontier model but STABLE run-to-run, which is what quality-delta R&D needs:
    the hosted Kimi judge drifts ~0.3 between sessions, so correctness was unmeasurable until
    this. recall@k stays the deterministic primary metric. Ollama serves an OpenAI-compatible
    API at /v1; api_key is a placeholder (ignored locally)."""
    client = ChatLLM(base_url=base_url, api_key="ollama", model=model)
    client._throttle = 0.0  # local: no rate limit
    client._seed = 0  # pin sampler for determinism
    client.max_workers = 1  # serialize: concurrent requests batch in Ollama and break determinism
    return client


if __name__ == "__main__":
    client = from_env()
    print(f"provider: {client.base_url}  model: {client.model}")
    print("reply:", client.complete("You are terse.", "Reply with the single word: pong").strip())
