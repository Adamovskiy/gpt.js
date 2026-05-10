import type { Tensor2d, Tensor3d } from '../tensorOps.ts';

import { Head } from './Head.ts';

export class MultiHeadAttention {
  readonly heads: Head[];
  readonly headSize: number;
  readonly numHeads: number;

  constructor(embeddingSize: number, numHeads: number) {
    this.numHeads = numHeads;
    this.headSize = Math.floor(embeddingSize / numHeads);
    this.heads = Array.from({ length: numHeads }, () => new Head(embeddingSize, this.headSize));
  }

  // x: (T, C), dOut: (T, C) -> gradients for all heads
  backward(
    x: Tensor2d,
    dOut: Tensor2d,
  ): {
    dX: Tensor2d;
    headGrads: { dWk: Tensor2d; dWq: Tensor2d; dWv: Tensor2d }[];
  } {
    // Split dOut back into per-head gradients: (T, numHeads * headSize) -> numHeads × (T, headSize)
    const dOutPerHead = this.heads.map((_, headIdx) =>
      dOut.map((tokenGrad) => tokenGrad.slice(headIdx * this.headSize, (headIdx + 1) * this.headSize)),
    );

    // Backward through each head
    const headResults = this.heads.map((head, headIdx) => head.backward(x, dOutPerHead[headIdx]));

    // Sum dX contributions from all heads
    const dX = headResults.reduce(
      (acc, { dX: headDX }) => acc.map((row, i) => row.map((val, j) => val + headDX[i][j])),
      x.map((row) => new Array<number>(row.length).fill(0)),
    );

    const headGrads = headResults.map(({ dWk, dWq, dWv }) => ({
      dWk,
      dWq,
      dWv,
    }));

    return { dX, headGrads };
  }

  // x: (B, T, C) -> (B, T, C)
  forward(x: Tensor3d): Tensor3d {
    const headOutputs = this.heads.map((head) => head.forward(x)); // Array of (B, T, headSize)

    // Concatenate along the last dimension: (B, T, numHeads * headSize)
    return x.map((_, batchIdx) =>
      x[batchIdx].map((_, tokenIdx) => headOutputs.flatMap((headOutput) => headOutput[batchIdx][tokenIdx])),
    );
  }
}
