import type { Tensor1d, Tensor2d, Tensor3d } from './tensorOps.js';

export interface Parameter {
  name: string;
  data: Tensor2d | Tensor1d;
}

export interface Trainable {
  forward(contextTokens: Tensor2d, targets?: Tensor2d): { logits: Tensor3d; loss?: number };
  getParameters(): Parameter[];
  computeGradients(contextTokens: Tensor2d, targets: Tensor2d): { [paramName: string]: Tensor2d | Tensor1d };
}
