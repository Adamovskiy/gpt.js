import { sum2d, sum3d, type Tensor1d, type Tensor2d, type Tensor3d } from '@/llm/tensorOps.ts';

import { FeedForward } from './FeedForward.ts';
import { LayerNorm } from './LayerNorm.ts';
import { MultiHeadAttention } from './MultiHeadAttention.ts';

export class TransformerBlock {
  readonly feedForward: FeedForward;
  readonly ln1: LayerNorm;
  readonly ln2: LayerNorm;
  readonly multiHeadAttention: MultiHeadAttention;

  constructor(embeddingSize: number, numHeads: number) {
    this.ln1 = new LayerNorm(embeddingSize);
    this.multiHeadAttention = new MultiHeadAttention(embeddingSize, numHeads);
    this.ln2 = new LayerNorm(embeddingSize);
    this.feedForward = new FeedForward(embeddingSize);
  }

  // x: (T, C), dOut: (T, C) -> gradients for all components
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
    // Forward pass to save intermediates
    const normed1 = this.ln1.forward([x])[0];
    const attended = this.multiHeadAttention.forward([normed1])[0];
    const afterAttn = sum2d(x, attended);
    const normed2 = this.ln2.forward([afterAttn])[0];
    // Backward pass: start from the end
    const dCurrent = dOut;

    // Gradient through second residual: afterFF = afterAttn + ffOut
    const dAfterAttn1 = dCurrent; // gradient flows through both branches
    const dFFOut = dCurrent;

    // Backward through FF
    const { dX: dNormed2, dW1, dB1, dW2, dB2 } = this.feedForward.backward(normed2, dFFOut);
    const ffGrads = { dW1, dB1, dW2, dB2 };

    // Backward through second LayerNorm
    const { dX: dAfterAttn2, dGamma: dGamma2, dBeta: dBeta2 } = this.ln2.backward(afterAttn, dNormed2);
    const ln2Grads = { dGamma: dGamma2, dBeta: dBeta2 };

    // Combine gradients flowing into afterAttn
    const dAfterAttn = sum2d(dAfterAttn1, dAfterAttn2);

    // Gradient through first residual: afterAttn = x + attended
    const dX1 = dAfterAttn;
    const dAttended = dAfterAttn;

    // Backward through MultiHeadAttention
    const { dX: dNormed1, headGrads } = this.multiHeadAttention.backward(normed1, dAttended);
    const attnGrads = headGrads;

    // Backward through first LayerNorm
    const { dX: dX2, dGamma: dGamma1, dBeta: dBeta1 } = this.ln1.backward(x, dNormed1);
    const ln1Grads = { dGamma: dGamma1, dBeta: dBeta1 };

    // Final gradient w.r.t. input
    const dX = sum2d(dX1, dX2);

    return { dX, ln1Grads, attnGrads, ln2Grads, ffGrads };
  }

  // x: (B, T, C) -> (B, T, C)
  forward(x: Tensor3d): Tensor3d {
    // Pre-norm architecture: norm → attention → residual → norm → ff → residual
    const normed1 = this.ln1.forward(x);
    const attended = this.multiHeadAttention.forward(normed1);
    const afterAttn = sum3d(x, attended); // x + attention(norm(x))

    const normed2 = this.ln2.forward(afterAttn);
    const ffOut = this.feedForward.forward(normed2);
    const afterFF = sum3d(afterAttn, ffOut); // afterAttn + ff(norm(afterAttn))

    return afterFF;
  }
}
