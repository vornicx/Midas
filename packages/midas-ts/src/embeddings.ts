/** Embeddings — the offline `HashingEmbedder` is a byte-for-byte port of the Python one
 * (same md5 token hashing, same sign/index math), so a TypeScript process produces the SAME
 * vectors as a Python process and both can share one on-disk store semantically.
 * `LocalEmbedder` adds local ONNX semantic embeddings via @huggingface/transformers (optional —
 * install it separately; everything degrades to the hashing embedder without it). */
import { createHash } from "node:crypto";

const WORD = /[\p{L}\p{N}_]+/gu;

export function tokenize(text: string): string[] {
  const out: string[] = [];
  for (const m of text.toLowerCase().matchAll(WORD)) {
    if (m[0].length > 2) out.push(m[0]);
  }
  return out;
}

export interface Embedder {
  dim: number;
  /** May be sync (HashingEmbedder) or async (LocalEmbedder) — callers `await` either. */
  embed(text: string): Float32Array | Promise<Float32Array>;
  embedMany(texts: string[]): Float32Array[] | Promise<Float32Array[]>;
}

export function l2Normalize(vec: Float32Array): Float32Array {
  let norm = 0;
  for (const v of vec) norm += v * v;
  norm = Math.sqrt(norm);
  if (norm === 0) return vec;
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] / norm;
  return out;
}

export function cosine(a: Float32Array, b: Float32Array): number {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

export class HashingEmbedder implements Embedder {
  readonly dim: number;

  constructor(dim = 256) {
    this.dim = dim;
  }

  embed(text: string): Float32Array {
    const vec = new Float64Array(this.dim);
    for (const tok of tokenize(text)) {
      const digest = createHash("md5").update(tok, "utf8").digest();
      const h = digest.readBigUInt64BE(0); // == Python int.from_bytes(digest[:8], "big")
      const idx = Number(h % BigInt(this.dim));
      const sign = (h >> 8n) & 1n ? 1.0 : -1.0;
      vec[idx] += sign;
    }
    // Normalize in float64 BEFORE casting to float32 — the same order Python uses, so the two
    // implementations produce bit-comparable vectors.
    let norm = 0;
    for (const v of vec) norm += v * v;
    norm = Math.sqrt(norm);
    if (norm > 0) for (let i = 0; i < vec.length; i++) vec[i] /= norm;
    return Float32Array.from(vec);
  }

  embedMany(texts: string[]): Float32Array[] {
    return texts.map((t) => this.embed(t));
  }
}

/** Local semantic embeddings over ONNX (no API key, nothing leaves the machine after the one-time
 * model download). Backed by `@huggingface/transformers`, which is NOT a dependency of this package
 * — install it yourself (`npm i @huggingface/transformers`) to enable `MIDAS_MCP_EMBEDDER=local`;
 * without it the server falls back to the offline hashing embedder and says so on stderr.
 * NOTE: vectors are model-specific. Don't point a bge-small (384-dim) process at a store written
 * with another embedder — keep one embedder per store file. */
export class LocalEmbedder implements Embedder {
  readonly dim: number;
  readonly model: string;
  private extractor: (text: string | string[], opts: object) => Promise<{ data: ArrayLike<number>; dims: number[] }>;

  private constructor(extractor: LocalEmbedder["extractor"], dim: number, model: string) {
    this.extractor = extractor;
    this.dim = dim;
    this.model = model;
  }

  static async create(model = "Xenova/bge-small-en-v1.5"): Promise<LocalEmbedder> {
    let transformers: { pipeline: (task: string, model: string, opts?: object) => Promise<unknown> };
    try {
      const spec = "@huggingface/transformers"; // variable specifier: optional dep, tsc must not resolve it
      transformers = await import(spec);
    } catch {
      throw new Error(
        "semantic embeddings need the optional `@huggingface/transformers` package — `npm i @huggingface/transformers`",
      );
    }
    const extractor = (await transformers.pipeline("feature-extraction", model, {
      dtype: "q8", // quantized: ~4x smaller download, near-identical retrieval
    })) as LocalEmbedder["extractor"];
    // bge-family models pool on the [CLS] token; probe once to learn the dimension.
    const probe = await extractor("dimension probe", { pooling: "cls", normalize: true });
    return new LocalEmbedder(extractor, probe.dims[probe.dims.length - 1], model);
  }

  async embed(text: string): Promise<Float32Array> {
    const out = await this.extractor(text, { pooling: "cls", normalize: true });
    return Float32Array.from(out.data as ArrayLike<number>);
  }

  async embedMany(texts: string[]): Promise<Float32Array[]> {
    const out: Float32Array[] = [];
    for (const t of texts) out.push(await this.embed(t));
    return out;
  }
}
