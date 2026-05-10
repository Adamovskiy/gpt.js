import type { Tensor2d, Tensor3d } from '../../tensorOps.ts';

import { GPUOperations } from '../../../gpu/gpuOps.ts';
import { HeadGPU } from './HeadGPU.ts';

export class MultiHeadAttentionGPU {
  readonly heads: HeadGPU[];
  readonly headSize: number;
  readonly numHeads: number;
  private device: GPUDevice | null = null;
  private gpuOps: GPUOperations | null = null;

  constructor(embeddingSize: number, numHeads: number) {
    this.numHeads = numHeads;
    this.headSize = Math.floor(embeddingSize / numHeads);
    this.heads = Array.from({ length: numHeads }, () => new HeadGPU(embeddingSize, this.headSize));
  }

  backward(
    x: Tensor2d,
    dOut: Tensor2d,
  ): {
    dX: Tensor2d;
    headGrads: { dWk: Tensor2d; dWq: Tensor2d; dWv: Tensor2d }[];
  } {
    const dOutPerHead = this.heads.map((_, headIdx) =>
      dOut.map((tokenGrad) => tokenGrad.slice(headIdx * this.headSize, (headIdx + 1) * this.headSize)),
    );

    const headResults = this.heads.map((head, headIdx) => head.backward(x, dOutPerHead[headIdx]));

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

  async forward(x: Tensor3d): Promise<Tensor3d> {
    const headOutputs = await Promise.all(this.heads.map((head) => head.forward(x)));

    return x.map((_, batchIdx) =>
      x[batchIdx].map((_, tokenIdx) => headOutputs.flatMap((headOutput) => headOutput[batchIdx][tokenIdx])),
    );
  }

  async initializeGPU(device: GPUDevice, gpuOps: GPUOperations): Promise<void> {
    this.device = device;
    this.gpuOps = gpuOps;
    for (const head of this.heads) {
      await head.initializeGPU(device, gpuOps);
    }
  }
}
