import type { Tensor1d } from '../tensorOps.ts';
import type { Tokenizer } from '../types.ts';

export interface CharTokenizerSerializedData {
  vocabulary: string[];
}

export class CharTokenizer implements Tokenizer<CharTokenizerSerializedData> {
  private readonly charToIndex: Map<string, number>;
  private readonly vocabulary: string[];

  constructor(fileContent: string) {
    this.vocabulary = Array.from(new Set(fileContent)).sort();
    this.charToIndex = new Map(this.vocabulary.map((ch, i) => [ch, i]));
  }

  static fromSerializedData(data: CharTokenizerSerializedData): CharTokenizer {
    return CharTokenizer.createFromData(data);
  }

  private static createFromData(data: CharTokenizerSerializedData): CharTokenizer {
    // Ignore read-only restriction when deserializing
    const instance = Object.create(CharTokenizer.prototype) as Record<string, unknown>;
    instance.vocabulary = data.vocabulary;
    instance.charToIndex = new Map(data.vocabulary.map((ch, i) => [ch, i]));
    return instance as unknown as CharTokenizer;
  }

  decode(indices: Tensor1d): string {
    return indices
      .map((i) => {
        if (i < 0 || i >= this.vocabulary.length) throw new Error(`Index out of range: ${i}`);
        return this.vocabulary[i];
      })
      .join('');
  }

  encode(str: string): Tensor1d {
    return Array.from(str).map((ch) => {
      const idx = this.charToIndex.get(ch);
      if (idx === undefined) throw new Error(`Unknown character: "${ch}"`);
      return idx;
    });
  }

  getSerializedData(): CharTokenizerSerializedData {
    return {
      vocabulary: this.vocabulary,
    };
  }

  getVocab(): string[] {
    return this.vocabulary;
  }

  getVocabSize() {
    return this.vocabulary.length;
  }
}
