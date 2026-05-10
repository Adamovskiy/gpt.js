import type { Tensor1d, Tensor2d, Tensor3d } from '../../tensorOps.ts';

import { GPUOperations } from '../../../gpu/gpuOps.ts';

export class LayerNormGPU {
  readonly beta: Tensor1d;
  readonly eps: number;
  readonly gamma: Tensor1d;
  private device: GPUDevice | null = null;
  private gpuOps: GPUOperations | null = null;

  constructor(embeddingSize: number, eps = 1e-5) {
    this.eps = eps;
    this.gamma = new Array<number>(embeddingSize).fill(1.0);
    this.beta = new Array<number>(embeddingSize).fill(0.0);
  }

  backward(
    x: Tensor2d,
    dOut: Tensor2d,
  ): {
    dBeta: Tensor1d;
    dGamma: Tensor1d;
    dX: Tensor2d;
  } {
    // CPU implementation for backward pass - same as LayerNorm
    const T = x.length;
    const C = x[0].length;

    const means = x.map((token) => token.reduce((sum, val) => sum + val, 0) / token.length);
    const variances = x.map((token, t) => token.reduce((sum, val) => sum + (val - means[t]) ** 2, 0) / token.length);
    const stds = variances.map((variance) => Math.sqrt(variance + this.eps));

    const dGamma = new Array<number>(C).fill(0);
    const dBeta = new Array<number>(C).fill(0);
    const dX = x.map((row) => new Array<number>(row.length).fill(0));

    for (let t = 0; t < T; t++) {
      const mean = means[t];
      const std = stds[t];

      for (let i = 0; i < C; i++) {
        const normalized = (x[t][i] - mean) / std;
        dGamma[i] += normalized * dOut[t][i];
        dBeta[i] += dOut[t][i];
      }

      const dNormalized = x[t].map((_, i) => this.gamma[i] * dOut[t][i]);
      const dVar = dNormalized.reduce((sum, dNorm, i) => sum + dNorm * (x[t][i] - mean), 0) * -0.5 * Math.pow(std, -3);
      const dMean =
        dNormalized.reduce((sum, dNorm) => sum + dNorm, 0) * (-1 / std) +
        dVar * x[t].reduce((sum, val) => sum + (val - mean), 0) * (-2 / C);

      for (let i = 0; i < C; i++) {
        dX[t][i] = dNormalized[i] / std + (dVar * 2 * (x[t][i] - mean)) / C + dMean / C;
      }
    }

    return { dX, dGamma, dBeta };
  }

  async forward(x: Tensor3d): Promise<Tensor3d> {
    if (!this.device || !this.gpuOps) {
      // Fallback to CPU implementation
      return x.map((batch) =>
        batch.map((token) => {
          const mean = token.reduce((sum, val) => sum + val, 0) / token.length;
          const variance = token.reduce((sum, val) => sum + (val - mean) ** 2, 0) / token.length;
          const std = Math.sqrt(variance + this.eps);
          return token.map((val, i) => ((val - mean) / std) * this.gamma[i] + this.beta[i]);
        }),
      );
    }

    // Use GPU implementation
    return await this.gpuOps.layerNorm(x, this.gamma, this.beta, this.eps);
  }

  async initializeGPU(device: GPUDevice, gpuOps: GPUOperations): Promise<void> {
    this.device = device;
    this.gpuOps = gpuOps;
  }
}
