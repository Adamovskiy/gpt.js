import { Linear } from '../models/Linear.ts';
import { type Tensor2d, type Tensor3d } from '../tensorOps.ts';

export interface Model {
  forward(contextTokens: Tensor2d, outputs: Tensor2d): { logits: Tensor3d; loss?: number };
  tokenEmbeddingTable: Tensor2d;
  languageModelingHead: Linear;
}

export interface Optimizer {
  train(contextTokens: Tensor2d, outputs: Tensor2d): Promise<number>;
}
