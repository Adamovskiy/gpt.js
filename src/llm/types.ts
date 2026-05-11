import type { Tensor1d, Tensor2d, Tensor3d } from './tensorOps.ts';

export interface Parameter {
  name: string;
  data: Tensor2d | Tensor1d;
}

export interface Serializable<T = unknown> {
  getSerializedData(): T;
}

export interface Tokenizer<T = unknown> extends Serializable<T> {
  encode(str: string): Tensor1d;
  decode(indices: Tensor1d): string;
  getVocabSize(): number;
  getVocab(): string[];
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

export interface LanguageModel<T = unknown> extends Serializable<T>, Trainable {
  generate(idx: Tensor2d, maxNewTokens: number): Promise<Tensor2d>;
}

export interface Optimizer<T = unknown> extends Serializable<T> {
  train(contextTokens: Tensor2d, outputs: Tensor2d): Promise<number>;
}
