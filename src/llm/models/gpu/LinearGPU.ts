import { GPUOperations } from '../../../gpu/gpuOps.ts';
import { random } from '../../../lib/random.ts';
import { matrixMultiply, sum1d, type Tensor1d, type Tensor3d } from '../../tensorOps.ts';

export class LinearGPU {
  readonly weights: number[][];
  readonly bias: number[];
  private device: GPUDevice | null = null;
  private gpuOps: GPUOperations | null = null;
  private weightsBuffer: GPUBuffer | null = null;
  private biasBuffer: GPUBuffer | null = null;
  private pipeline: GPUComputePipeline | null = null;

  constructor(inputSize: number, outputSize: number, useBias = true) {
    this.weights = Array.from({ length: inputSize }, () => Array.from({ length: outputSize }, () => random() * 0.01));
    this.bias = useBias
      ? Array.from({ length: outputSize }, () => random() * 0.01)
      : new Array<number>(outputSize).fill(0);
  }

  async initializeGPU(device: GPUDevice, gpuOps: GPUOperations): Promise<void> {
    this.device = device;
    this.gpuOps = gpuOps;
    // GPU buffer initialization would go here
    // For now, keeping it simple
  }

  // CPU version for compatibility
  forward(input: Tensor1d): Tensor1d {
    return sum1d(matrixMultiply([input], this.weights)[0], this.bias);
  }

  // GPU-accelerated version for batched operations
  async forwardBatched(input: Tensor3d): Promise<Tensor3d> {
    if (!this.device || !this.gpuOps) {
      // Fallback to CPU implementation
      return input.map((batch) => batch.map((token) => this.forward(token)));
    }

    // GPU implementation using matmul for each batch
    const result: Tensor3d = [];
    for (let b = 0; b < input.length; b++) {
      const batchResult = await this.gpuOps.matrixMultiply(input[b], this.weights);
      // Add bias if needed
      result[b] = batchResult.map((token) => sum1d(token, this.bias));
    }

    return result;
  }
}
