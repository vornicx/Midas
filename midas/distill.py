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
        return [
            line.strip(" -•\t")
            for line in out.splitlines()
            if line.strip() and not line.strip().lower().startswith(("facts:", "here are"))
        ]
