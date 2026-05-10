import type { Tensor1d } from '../tensorOps.ts';
import type { Tokenizer } from '../types.ts';

/*
Byte Pair Encoding – individual characters + most popular pairs
 */
export interface BPETokenizerSerializedData {
  vocabulary: string[];
  mergeRules: [string, string][];
}

export class BPETokenizer implements Tokenizer<BPETokenizerSerializedData> {
  private readonly mergeRules: [string, string][];
  private readonly tokenToIndex: Map<string, number>;
  private readonly vocabulary: string[];

  constructor(fileContent: string, numMerges: number, onProgress: (progress: number) => void) {
    // Get unique characters first
    const chars = Array.from(new Set(fileContent)).sort();
    this.vocabulary = [...chars];
    this.mergeRules = [];

    onProgress(1 / (numMerges + 1));

    // Simple approach: just find most frequent pairs and remember them
    // Don't modify the original text during training

    for (let merge = 0; merge < numMerges; merge++) {
      const pairCounts = new Map<string, number>();

      // Count pairs in original text, ignoring whitespace pairs
      for (let i = 0; i < fileContent.length - 1; i++) {
        const char1 = fileContent[i];
        const char2 = fileContent[i + 1];

        // Skip whitespace pairs
        if (char1 !== ' ' && char2 !== ' ' && char1 !== '\n' && char2 !== '\n') {
          const pair = char1 + char2;
          pairCounts.set(pair, (pairCounts.get(pair) ?? 0) + 1);
        }
      }

      if (pairCounts.size === 0) break;

      // Find most frequent pair that's not already in vocab
      let bestPair = '';
      let maxCount = 0;
      for (const [pair, count] of pairCounts.entries()) {
        if (count > maxCount && !this.vocabulary.includes(pair)) {
          maxCount = count;
          bestPair = pair;
        }
      }

      if (maxCount < 3 || bestPair === '') break; // Need at least 3 occurrences

      // Add to vocabulary and remember the rule
      this.vocabulary.push(bestPair);
      this.mergeRules.push([bestPair[0], bestPair[1]]);
      onProgress((2 + merge) / (numMerges + 1));
    }

    // Create token to index mapping
    this.tokenToIndex = new Map(this.vocabulary.map((token, i) => [token, i]));
  }

  static fromSerializedData(data: BPETokenizerSerializedData): BPETokenizer {
    return BPETokenizer.createFromData(data);
  }

  private static createFromData(data: BPETokenizerSerializedData): BPETokenizer {
    // Ignore read-only restriction when deserializing
    const instance = Object.create(BPETokenizer.prototype) as Record<string, unknown>;
    instance.vocabulary = data.vocabulary;
    instance.mergeRules = data.mergeRules;
    instance.tokenToIndex = new Map(data.vocabulary.map((token, i) => [token, i]));
    return instance as unknown as BPETokenizer;
  }

  decode(indices: Tensor1d): string {
    return indices
      .map((i) => {
        if (i < 0 || i >= this.vocabulary.length) {
          throw new Error(`Index out of range: ${i}`);
        }
        return this.vocabulary[i];
      })
      .join('');
  }

  encode(str: string): Tensor1d {
    // Start with individual characters
    let tokens = Array.from(str);

    // Apply merge rules in the order they were learned
    for (const [first, second] of this.mergeRules) {
      const newTokens: string[] = [];
      let i = 0;

      while (i < tokens.length) {
        if (i < tokens.length - 1 && tokens[i] === first && tokens[i + 1] === second) {
          // Merge these tokens
          newTokens.push(first + second);
          i += 2;
        } else {
          newTokens.push(tokens[i]);
          i += 1;
        }
      }

      tokens = newTokens;
    }

    // Convert to indices
    const result: number[] = [];
    for (const token of tokens) {
      const idx = this.tokenToIndex.get(token);
      if (idx === undefined) {
        throw new Error(`Unknown token: "${token}"`);
      }
      result.push(idx);
    }

    return result;
  }

  getSerializedData(): BPETokenizerSerializedData {
    return {
      vocabulary: this.vocabulary,
      mergeRules: this.mergeRules,
    };
  }

  getVocab(): string[] {
    return this.vocabulary;
  }

  getVocabSize(): number {
    return this.vocabulary.length;
  }
}
