import { GPUOperations } from '../../../gpu/gpuOps.ts';
import { matrixMultiply, type Tensor1d, type Tensor2d, type Tensor3d, transpose } from '../../tensorOps.ts';
import { LinearGPU } from './LinearGPU.ts';

export class FeedForwardGPU {
  readonly linear1: LinearGPU;
  readonly linear2: LinearGPU;
  private device: GPUDevice | null = null;
  private gpuOps: GPUOperations | null = null;

  constructor(embeddingSize: number, hiddenSize?: number) {
    const ffnDim = hiddenSize || 4 * embeddingSize;
    this.linear1 = new LinearGPU(embeddingSize, ffnDim);
    this.linear2 = new LinearGPU(ffnDim, embeddingSize);
  }

  backward(
    x: Tensor2d,
    dOut: Tensor2d,
  ): {
    dB1: Tensor1d;
    dB2: Tensor1d;
    dW1: Tensor2d;
    dW2: Tensor2d;
    dX: Tensor2d;
  } {
    // CPU implementation - same as FeedForward
    const T = x.length;

    const hidden = x.map((token) => this.linear1.forward(token));
    const activated = hidden.map((h) => h.map((val) => Math.max(0, val)));

    const dW1 = this.linear1.weights.map((row) => new Array<number>(row.length).fill(0));
    const dB1 = new Array<number>(this.linear1.bias.length).fill(0);
    const dW2 = this.linear2.weights.map((row) => new Array<number>(row.length).fill(0));
    const dB2 = new Array<number>(this.linear2.bias.length).fill(0);

    const dActivated = matrixMultiply(dOut, transpose(this.linear2.weights));

    for (let t = 0; t < T; t++) {
      for (let i = 0; i < activated[t].length; i++) {
        for (let j = 0; j < dOut[t].length; j++) {
          dW2[i][j] += activated[t][i] * dOut[t][j];
        }
      }
      for (let j = 0; j < dOut[t].length; j++) {
        dB2[j] += dOut[t][j];
      }
    }

    const dHidden = dActivated.map((row, t) => row.map((grad, i) => (hidden[t][i] > 0 ? grad : 0)));
    const dX = matrixMultiply(dHidden, transpose(this.linear1.weights));

    for (let t = 0; t < T; t++) {
      for (let i = 0; i < x[t].length; i++) {
        for (let j = 0; j < dHidden[t].length; j++) {
          dW1[i][j] += x[t][i] * dHidden[t][j];
        }
      }
      for (let j = 0; j < dHidden[t].length; j++) {
        dB1[j] += dHidden[t][j];
      }
    }

    return { dX, dW1, dB1, dW2, dB2 };
  }

  async forward(x: Tensor3d): Promise<Tensor3d> {
    if (!this.device) {
      // Fallback to CPU implementation
      return x.map((batch) =>
        batch.map((token) => {
          const hidden = this.linear1.forward(token);
          const activated = hidden.map((val) => Math.max(0, val)); // ReLU
          return this.linear2.forward(activated);
        }),
      );
    }

    // GPU implementation would go here
    return x.map((batch) =>
      batch.map((token) => {
        const hidden = this.linear1.forward(token);
        const activated = hidden.map((val) => Math.max(0, val)); // ReLU
        return this.linear2.forward(activated);
      }),
    );
  }

  async initializeGPU(device: GPUDevice, gpuOps: GPUOperations): Promise<void> {
    this.device = device;
    this.gpuOps = gpuOps;
    await this.linear1.initializeGPU(device, gpuOps);
    await this.linear2.initializeGPU(device, gpuOps);
  }
}
