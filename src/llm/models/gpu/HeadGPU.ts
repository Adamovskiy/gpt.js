import { GPUOperations } from '../../../gpu/gpuOps.ts';
import {
  matrixMultiply,
  softmax,
  softmaxBackward,
  sum2d,
  type Tensor2d,
  type Tensor3d,
  transpose,
} from '../../tensorOps.ts';
import { LinearGPU } from './LinearGPU.ts';

export class HeadGPU {
  readonly headSize: number;
  readonly key: LinearGPU;
  readonly query: LinearGPU;
  readonly value: LinearGPU;
  private device: GPUDevice | null = null;
  private gpuOps: GPUOperations | null = null;

  constructor(embeddingSize: number, headSize: number) {
    this.headSize = headSize;
    this.key = new LinearGPU(embeddingSize, headSize, false);
    this.query = new LinearGPU(embeddingSize, headSize, false);
    this.value = new LinearGPU(embeddingSize, headSize, false);
  }

  backward(
    x: Tensor2d,
    dOut: Tensor2d,
  ): {
    dWk: Tensor2d;
    dWq: Tensor2d;
    dWv: Tensor2d;
    dX: Tensor2d;
  } {
    // CPU implementation - same as Head.backward
    const scale = Math.pow(x[0].length, -0.5);

    const k = x.map((token) => this.key.forward(token));
    const q = x.map((token) => this.query.forward(token));
    const v = x.map((token) => this.value.forward(token));
    const wei = matrixMultiply(q, transpose(k)).map((row) => row.map((w) => w * scale));
    const maskedWei = wei.map((row, i) => row.map((w, j) => (j <= i ? w : -Infinity)));
    const weightedWei = maskedWei.map(softmax);

    const dWeightedWei = matrixMultiply(dOut, transpose(v));
    const dV = matrixMultiply(transpose(weightedWei), dOut);

    const dWei = weightedWei.map((row, i) => softmaxBackward(row, dWeightedWei[i]));

    const dQ = matrixMultiply(dWei, k).map((row) => row.map((w) => w * scale));
    const dK = matrixMultiply(transpose(dWei), q).map((row) => row.map((w) => w * scale));

    const dWk = matrixMultiply(transpose(x), dK);
    const dWq = matrixMultiply(transpose(x), dQ);
    const dWv = matrixMultiply(transpose(x), dV);

    const dX = sum2d(
      sum2d(matrixMultiply(dQ, transpose(this.query.weights)), matrixMultiply(dK, transpose(this.key.weights))),
      matrixMultiply(dV, transpose(this.value.weights)),
    );

    return { dX, dWk, dWq, dWv };
  }

  async forward(x: Tensor3d): Promise<Tensor3d> {
    if (!this.device) {
      // Fallback to CPU implementation - same as Head.forward
      const embeddingSize = x[0][0].length;
      const scale = Math.pow(embeddingSize, -0.5);

      return x.map((batch) => {
        const k = batch.map((token) => this.key.forward(token));
        const q = batch.map((token) => this.query.forward(token));
        const v = batch.map((token) => this.value.forward(token));

        const wei: Tensor2d = matrixMultiply(q, transpose(k)).map((row) => row.map((w) => w * scale));
        const maskedWei: Tensor2d = wei.map((row, i) => row.map((w, j) => (j <= i ? w : -Infinity)));
        const weightedWei = maskedWei.map(softmax);

        return matrixMultiply(weightedWei, v);
      });
    }

    // GPU implementation would go here
    return x.map((batch) => {
      const k = batch.map((token) => this.key.forward(token));
      const q = batch.map((token) => this.query.forward(token));
      const v = batch.map((token) => this.value.forward(token));

      const embeddingSize = x[0][0].length;
      const scale = Math.pow(embeddingSize, -0.5);
      const wei: Tensor2d = matrixMultiply(q, transpose(k)).map((row) => row.map((w) => w * scale));
      const maskedWei: Tensor2d = wei.map((row, i) => row.map((w, j) => (j <= i ? w : -Infinity)));
      const weightedWei = maskedWei.map(softmax);

      return matrixMultiply(weightedWei, v);
    });
  }

  async initializeGPU(device: GPUDevice, gpuOps: GPUOperations): Promise<void> {
    this.device = device;
    this.gpuOps = gpuOps;
    await this.key.initializeGPU(device, gpuOps);
    await this.query.initializeGPU(device, gpuOps);
    await this.value.initializeGPU(device, gpuOps);
  }
}
