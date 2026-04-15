import type { Tensor1d } from './tensorOps.js';

export class CharTokenizer {
  private readonly vocabulary: string[];
  private readonly charToIndex: Map<string, number>;

  constructor(fileContent: string) {
    this.vocabulary = Array.from(new Set(fileContent)).sort();
    this.charToIndex = new Map(this.vocabulary.map((ch, i) => [ch, i]));
  }

  encode(str: string): Tensor1d {
    return Array.from(str).map((ch) => {
      const idx = this.charToIndex.get(ch);
      if (idx === undefined) throw new Error(`Unknown character: "${ch}"`);
      return idx;
    });
  }

  decode(indices: Tensor1d): string {
    return indices
      .map((i) => {
        if (i < 0 || i >= this.vocabulary.length) throw new Error(`Index out of range: ${i}`);
        return this.vocabulary[i];
      })
      .join('');
  }

  getVocabSize() {
    return this.vocabulary.length;
  }
}
