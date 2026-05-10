import type { Tensor1d, Tensor2d, Tensor3d } from './tensorOps.ts';

export interface Parameter {
  name: string;
  data: Tensor2d | Tensor1d;
}

export interface Tokenizer<T = unknown> {
  encode(str: string): Tensor1d;
  decode(indices: Tensor1d): string;
  getVocabSize(): number;
  getVocab(): string[];
  getSerializedData(): T;
}

export interface Trainable {
  forward(contextTokens: Tensor2d, targets?: Tensor2d): Promise<{ logits: Tensor3d; loss?: number }>;
  getParameters(): Parameter[];
  computeGradients(
    contextTokens: Tensor2d,
    targets: Tensor2d,
    precomputedLogits?: Tensor3d,
  ): Record<string, Tensor2d | Tensor1d>;
}

export interface LanguageModel extends Trainable {
  generate(idx: Tensor2d, maxNewTokens: number): Promise<Tensor2d>;
}
