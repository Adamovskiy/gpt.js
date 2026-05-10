import type { Tensor1d, Tensor2d } from './tensorOps.ts';

import { random } from '../lib/random.ts';

export const blockSize = 8; // Max context length for predictions
const batchSize = 4; // Independent sequences processed in parallel

export function getBatch(data: Tensor1d): {
  contexts: Tensor2d;
  outputs: Tensor2d;
} {
  if (data.length < blockSize + 1) throw new Error(`Data is to small for blockSize ${blockSize}`);

  const contexts: Tensor2d = [];
  const outputs: Tensor2d = [];

  for (let batchIdx = 0; batchIdx < batchSize; batchIdx++) {
    const offset = Math.floor(random() * (data.length - blockSize));

    contexts.push(data.slice(offset, offset + blockSize));
    outputs.push(data.slice(offset + 1, offset + blockSize + 1));
  }

  // Two batchSize x blockSize 2D tensors
  return {
    contexts,
    outputs,
  };
}
