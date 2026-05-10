import { Linear } from './Linear.ts';
import { matrixMultiply, type Tensor1d, type Tensor2d, type Tensor3d, transpose } from '../tensorOps.ts';

export class FeedForward {
  readonly linear1: Linear;
  readonly linear2: Linear;

  constructor(embeddingSize: number, hiddenSize?: number) {
    const ffnDim = hiddenSize || 4 * embeddingSize; // Standard transformer ratio
    this.linear1 = new Linear(embeddingSize, ffnDim);
    this.linear2 = new Linear(ffnDim, embeddingSize);
  }

  // x: (B, T, C) -> (B, T, C)
  forward(x: Tensor3d): Tensor3d {
    return x.map((batch) =>
      batch.map((token) => {
        const hidden = this.linear1.forward(token);
        const activated = hidden.map((val) => Math.max(0, val)); // ReLU activation
        return this.linear2.forward(activated);
      }),
    );
  }

  // x: (T, C), dOut: (T, C) -> gradients w.r.t. x and weight matrices
  backward(
    x: Tensor2d,
    dOut: Tensor2d,
  ): {
    dX: Tensor2d;
    dW1: Tensor2d;
    dB1: Tensor1d;
    dW2: Tensor2d;
    dB2: Tensor1d;
  } {
    const T = x.length;

    // Forward pass to get intermediate values
    const hidden = x.map((token) => this.linear1.forward(token)); // (T, hiddenSize)
    const activated = hidden.map((h) => h.map((val) => Math.max(0, val))); // ReLU

    // Initialize gradients
    const dW1 = this.linear1.weights.map((row) => new Array<number>(row.length).fill(0));
    const dB1 = new Array<number>(this.linear1.bias.length).fill(0);
    const dW2 = this.linear2.weights.map((row) => new Array<number>(row.length).fill(0));
    const dB2 = new Array<number>(this.linear2.bias.length).fill(0);

    // Backward through second linear layer: out = activated @ W2 + b2
    const dActivated = matrixMultiply(dOut, transpose(this.linear2.weights)); // (T, hiddenSize)

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

    // Backward through ReLU: derivative is 1 if input > 0, else 0
    const dHidden = dActivated.map((row, t) => row.map((grad, i) => (hidden[t][i] > 0 ? grad : 0))); // (T, hiddenSize)

    // Backward through first linear layer: hidden = x @ W1 + b1
    const dX = matrixMultiply(dHidden, transpose(this.linear1.weights)); // (T, C)

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
}
