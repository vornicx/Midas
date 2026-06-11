/** BM25 lexical ranking — port of `midas/bm25.py` (Okapi BM25, non-negative IDF, per-term
 * posting lists so a query touches only documents sharing a term with it). */
import { tokenize } from "./embeddings.js";
import type { MemoryRecord } from "./types.js";

const K1 = 1.5;
const B = 0.75;

export class BM25 {
  private readonly records: MemoryRecord[];
  private readonly dls: number[] = [];
  private readonly postings = new Map<string, Array<[number, number]>>();
  private readonly idf = new Map<string, number>();
  private readonly avgdl: number;

  constructor(records: MemoryRecord[]) {
    this.records = records;
    for (let i = 0; i < records.length; i++) {
      const doc = tokenize(records[i].content);
      this.dls.push(doc.length);
      const tf = new Map<string, number>();
      for (const t of doc) tf.set(t, (tf.get(t) ?? 0) + 1);
      for (const [term, f] of tf) {
        let list = this.postings.get(term);
        if (!list) this.postings.set(term, (list = []));
        list.push([i, f]);
      }
    }
    this.avgdl = this.dls.length ? this.dls.reduce((a, b) => a + b, 0) / this.dls.length : 0;
    const n = records.length;
    for (const [t, p] of this.postings) {
      this.idf.set(t, Math.log(1 + (n - p.length + 0.5) / (p.length + 0.5)));
    }
  }

  /** Map record id -> BM25 score (only positive scores included). */
  scores(query: string): Map<string, number> {
    const qTerms = tokenize(query);
    const avgdl = this.avgdl || 1.0;
    const byDoc = new Map<number, number>();
    for (const term of qTerms) {
      const postings = this.postings.get(term);
      if (!postings) continue;
      const idf = this.idf.get(term)!;
      for (const [i, f] of postings) {
        const dl = this.dls[i];
        const s = (byDoc.get(i) ?? 0) + (idf * (f * (K1 + 1))) / (f + K1 * (1 - B + (B * dl) / avgdl));
        byDoc.set(i, s);
      }
    }
    const out = new Map<string, number>();
    for (const [i, s] of byDoc) if (s > 0) out.set(this.records[i].id, s);
    return out;
  }
}
