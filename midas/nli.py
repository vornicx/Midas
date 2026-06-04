"""Local natural-language-inference (entailment / contradiction) — no LLM, no API, no torch.

Runs a small MNLI cross-encoder (ONNX via onnxruntime + tokenizers, the same local stack as
`LocalEmbedder`/`LocalReranker`). Two uses in Midas, both LLM-free:

  - **Calibrated / abstention**: does any retrieved turn *entail* an answer to the question? If none
    do, the answer isn't in memory -> abstain instead of confabulating. This tests answer-PRESENCE,
    which embedding cosine and even a relevance reranker conflate with mere topic-match.
  - **Current / belief revision**: does a new turn *contradict* an existing belief? If so, supersede
    it. Far more precise than a keyword/cue heuristic (which fires on distinct-but-similar facts).

Default model: `Xenova/distilbert-base-uncased-mnli` (~67M, int8 ONNX ~70MB) — labels
ENTAILMENT / NEUTRAL / CONTRADICTION. Lazy-loaded; first use downloads the model.
"""
from __future__ import annotations

import json

from .embeddings import configure_local_model_cache

_DEFAULT_MODEL = "Xenova/distilbert-base-uncased-mnli"


class LocalNLI:
    """Cheap local entailment scorer. Stateless per call; the model loads once and is reused."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        *,
        cache_dir: str | None = None,
        max_length: int = 256,
        quantized: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.cache_dir = configure_local_model_cache(cache_dir)
        self._quantized = quantized
        self._sess = None
        self._tok = None
        self._order: list[str] = []  # softmax index -> label name (from config id2label)

    def _ensure(self) -> None:
        if self._sess is not None:
            return
        import numpy as np  # noqa: F401  (used in scores)
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        onnx_file = "onnx/model_quantized.onnx" if self._quantized else "onnx/model.onnx"
        onnx_path = hf_hub_download(self.model_name, onnx_file, cache_dir=self.cache_dir)
        tok_path = hf_hub_download(self.model_name, "tokenizer.json", cache_dir=self.cache_dir)
        cfg = json.load(open(hf_hub_download(self.model_name, "config.json", cache_dir=self.cache_dir)))
        id2label = {int(k): v.upper() for k, v in cfg.get("id2label", {}).items()}
        self._order = [id2label.get(i, str(i)) for i in range(len(id2label))]
        self._tok = Tokenizer.from_file(tok_path)
        self._tok.enable_truncation(max_length=self.max_length)
        # Single-threaded session: small model, called per-pair; avoids oversubscribing cores.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self._sess = ort.InferenceSession(onnx_path, sess_options=opts)
        self._input_names = {i.name for i in self._sess.get_inputs()}

    def scores(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Return {'ENTAILMENT': p, 'NEUTRAL': p, 'CONTRADICTION': p} for premise -> hypothesis."""
        self._ensure()
        import numpy as np

        enc = self._tok.encode(premise, hypothesis)
        feeds = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)
        logits = self._sess.run(None, feeds)[0][0]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        return {label: float(p) for label, p in zip(self._order, probs)}

    def entailment(self, premise: str, hypothesis: str) -> float:
        return self.scores(premise, hypothesis).get("ENTAILMENT", 0.0)

    def contradiction(self, a: str, b: str) -> float:
        """Max contradiction across both directions (contradiction is symmetric in intent)."""
        return max(
            self.scores(a, b).get("CONTRADICTION", 0.0),
            self.scores(b, a).get("CONTRADICTION", 0.0),
        )
