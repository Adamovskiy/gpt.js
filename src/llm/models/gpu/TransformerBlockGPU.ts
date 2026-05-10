import { GPUOperations } from '../../../gpu/gpuOps.ts';
import { sum2d, sum3d, type Tensor1d, type Tensor2d, type Tensor3d } from '../../tensorOps.ts';
import { FeedForwardGPU } from './FeedForwardGPU.ts';
import { LayerNormGPU } from './LayerNormGPU.ts';
import { MultiHeadAttentionGPU } from './MultiHeadAttentionGPU.ts';

export class TransformerBlockGPU {
  readonly feedForward: FeedForwardGPU;
  readonly ln1: LayerNormGPU;
  readonly ln2: LayerNormGPU;
  readonly multiHeadAttention: MultiHeadAttentionGPU;
  private device: GPUDevice | null = null;
  private gpuOps: GPUOperations | null = null;

  constructor(embeddingSize: number, numHeads: number) {
    this.ln1 = new LayerNormGPU(embeddingSize);
    this.multiHeadAttention = new MultiHeadAttentionGPU(embeddingSize, numHeads);
    this.ln2 = new LayerNormGPU(embeddingSize);
    this.feedForward = new FeedForwardGPU(embeddingSize);
  }

  backward(
    x: Tensor2d,
    dOut: Tensor2d,
  ): {
    attnGrads: { dWk: Tensor2d; dWq: Tensor2d; dWv: Tensor2d }[];
    dX: Tensor2d;
    ffGrads: { dB1: Tensor1d; dB2: Tensor1d; dW1: Tensor2d; dW2: Tensor2d };
    ln1Grads: { dBeta: Tensor1d; dGamma: Tensor1d };
    ln2Grads: { dBeta: Tensor1d; dGamma: Tensor1d };
  } {
    // CPU implementation - same as TransformerBlock.backward
    const normed1 = this.ln1.forward([x])[0];
    const attended = this.multiHeadAttention.forward([normed1])[0];
    const afterAttn = sum2d(x, attended);
    const normed2 = this.ln2.forward([afterAttn])[0];

    const dCurrent = dOut;
    const dAfterAttn1 = dCurrent;
    const dFFOut = dCurrent;

    const { dX: dNormed2, dW1, dB1, dW2, dB2 } = this.feedForward.backward(normed2, dFFOut);
    const ffGrads = { dW1, dB1, dW2, dB2 };

    const { dX: dAfterAttn2, dGamma: dGamma2, dBeta: dBeta2 } = this.ln2.backward(afterAttn, dNormed2);
    const ln2Grads = { dGamma: dGamma2, dBeta: dBeta2 };

    const dAfterAttn = sum2d(dAfterAttn1, dAfterAttn2);
    const dX1 = dAfterAttn;
    const dAttended = dAfterAttn;

    const { dX: dNormed1, headGrads } = this.multiHeadAttention.backward(normed1, dAttended);
    const attnGrads = headGrads;

    const { dX: dX2, dGamma: dGamma1, dBeta: dBeta1 } = this.ln1.backward(x, dNormed1);
    const ln1Grads = { dGamma: dGamma1, dBeta: dBeta1 };

    const dX = sum2d(dX1, dX2);

    return { dX, ln1Grads, attnGrads, ln2Grads, ffGrads };
  }

  async forward(x: Tensor3d): Promise<Tensor3d> {
    if (!this.gpuOps) {
      // Fallback to CPU implementation
      const normed1 = await this.ln1.forward(x);
      const attended = await this.multiHeadAttention.forward(normed1);
      const afterAttn = sum3d(x, attended);

      const normed2 = await this.ln2.forward(afterAttn);
      const ffOut = await this.feedForward.forward(normed2);
      const afterFF = sum3d(afterAttn, ffOut);

      return afterFF;
    }

    // GPU-accelerated implementation
    const normed1 = await this.ln1.forward(x);
    const attended = await this.multiHeadAttention.forward(normed1);
    const afterAttn = await this.gpuOps.elementwiseAdd(x, attended);

    const normed2 = await this.ln2.forward(afterAttn);
    const ffOut = await this.feedForward.forward(normed2);
    const afterFF = await this.gpuOps.elementwiseAdd(afterAttn, ffOut);

    return afterFF;
  }

  async initializeGPU(device: GPUDevice, gpuOps: GPUOperations): Promise<void> {
    this.device = device;
    this.gpuOps = gpuOps;
    await this.ln1.initializeGPU(device, gpuOps);
    await this.multiHeadAttention.initializeGPU(device, gpuOps);
    await this.ln2.initializeGPU(device, gpuOps);
    await this.feedForward.initializeGPU(device, gpuOps);
  }
}
