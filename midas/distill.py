"""Optional distillation — the explicitly-fenced "crosses the no-LLM line" tier.

Midas's default tier (raw turns) and agent-driven tier (the agent's own LLM distills via `capture`)
never call an LLM inside Midas. This module is the THIRD tier, for **non-agentic ingest**: raw logs
or turns with no smart agent in the loop, where you still want compact, high-signal facts but have
no agent LLM to delegate to. Plug in YOUR model — ideally a **local** one (Ollama), so the
$0/local/zero-egress properties hold even though an LLM now runs at ingest.

The honest tradeoff this tier makes — explicit and opt-in, never the default:
  - **adds** automatic distillation (the Mem0/Letta/LIGHT ingest behaviour), and
  - **gives up** verbatim source-traceability of the distilled record (it is LLM-rewritten, not a
    source turn) and determinism (an LLM at ingest is not reproducible).
Keep the raw turns alongside (`keep_raw=True`) to retain the audit trail; the distilled facts are an
index layer on top, not a replacement.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Distiller(Protocol):
    """Turn a batch of raw turns/texts into compact, self-contained fact strings (one per item)."""

    def distill(self, texts: list[str]) -> list[str]: ...


DISTILL_PROMPT = (
    "Extract the durable, reusable knowledge from these conversation turns. For each item, write "
    "ONE compact, self-contained sentence — name the entities, the value, and when it holds — so it "
    "answers on its own without the surrounding chat. Output one fact per line: no numbering, no "
    "commentary, no preamble. Skip pure small talk.\n\nTurns:\n{turns}\n\nFacts:"
)


class OllamaDistiller:
    """Reference **local** distiller via Ollama — $0, on-device, zero data egress (the moat holds)
    while adding automatic distillation. Needs `ollama serve` and a pulled model (e.g.
    `ollama pull llama3.2:3b`). Uses only stdlib (`urllib`), no new dependency. Non-deterministic
    by nature (it is an LLM), so the distilled facts are NOT verbatim sources — the tradeoff this
    tier makes explicit."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
        *,
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def distill(self, texts: list[str]) -> list[str]:
        import json
        import urllib.request

        if not texts:
            return []
        prompt = DISTILL_PROMPT.format(turns="\n".join(texts))
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read())["response"]
        return _parse_facts(out)


class HTTPDistiller:
    """Distiller over any **OpenAI-compatible** `/chat/completions` endpoint — a local vLLM/LM Studio
    (keeps $0/local/zero-egress), or a cloud gateway (OpenRouter, OpenAI…) which trades the moat for
    quality. Reads the key from `api_key` or `api_key_env`. stdlib only (`urllib`)."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        api_key_env: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        import os

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None) or "none"
        self.temperature = temperature
        self.timeout = timeout

    def distill(self, texts: list[str]) -> list[str]:
        import json
        import urllib.request

        if not texts:
            return []
        prompt = DISTILL_PROMPT.format(turns="\n".join(texts))
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": 1024,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read())["choices"][0]["message"]["content"]
        return _parse_facts(out)


def _parse_facts(out: str) -> list[str]:
    return [
        line.strip(" -•\t0123456789.")
        for line in out.splitlines()
        if line.strip() and not line.strip().lower().startswith(("facts:", "here are", "based on"))
    ]
