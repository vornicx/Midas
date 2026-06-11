/** Embeddings — the offline `HashingEmbedder` is a byte-for-byte port of the Python one
 * (same md5 token hashing, same sign/index math), so a TypeScript process produces the SAME
 * vectors as a Python process and both can share one on-disk store semantically.
 * A local ONNX semantic embedder (bge / multilingual) is the planned next step. */
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
  embed(text: string): Float32Array;
  embedMany(texts: string[]): Float32Array[];
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
