import {
  matrixMultiply,
  softmax,
  softmaxBackward,
  sum2d,
  type Tensor2d,
  type Tensor3d,
  transpose,
} from '../tensorOps.ts';
import { Linear } from './Linear.ts';

export class Head {
  readonly headSize: number;
  readonly key: Linear; // projects embedding -> headSize (no bias, as in nanoGPT)
  readonly query: Linear;

  readonly value: Linear;

  constructor(embeddingSize: number, headSize: number) {
    this.headSize = headSize;
    this.key = new Linear(embeddingSize, headSize, false);
    this.query = new Linear(embeddingSize, headSize, false);
    this.value = new Linear(embeddingSize, headSize, false);
  }

  // x: (T, C), dOut: (T, headSize) -> gradients w.r.t. x and weight matrices
  backward(
    x: Tensor2d,
    dOut: Tensor2d,
  ): {
    dWk: Tensor2d;
    dWq: Tensor2d;
    dWv: Tensor2d;
    dX: Tensor2d;
  } {
    const scale = Math.pow(x[0].length, -0.5);

    // Recompute forward pass values needed for backward
    const k = x.map((token) => this.key.forward(token)); // (T, H)
    const q = x.map((token) => this.query.forward(token)); // (T, H)
    const v = x.map((token) => this.value.forward(token)); // (T, H)
    const wei = matrixMultiply(q, transpose(k)).map((row) => row.map((w) => w * scale)); // (T, T)
    const maskedWei = wei.map((row, i) => row.map((w, j) => (j <= i ? w : -Infinity))); // (T, T)
    const weightedWei = maskedWei.map(softmax); // (T, T)

    // Backward through: out = weightedWei @ v
    const dWeightedWei = matrixMultiply(dOut, transpose(v)); // (T, T)
    const dV = matrixMultiply(transpose(weightedWei), dOut); // (T, H)

    // Backward through row-wise softmax (upper triangle stays 0 naturally)
    const dWei = weightedWei.map((row, i) => softmaxBackward(row, dWeightedWei[i])); // (T, T)

    // Backward through: wei = q @ k^T * scale
    const dQ = matrixMultiply(dWei, k).map((row) => row.map((w) => w * scale)); // (T, H)
    const dK = matrixMultiply(transpose(dWei), q).map((row) => row.map((w) => w * scale)); // (T, H)

    // Backward through linear projections (no bias): dW = x^T @ d_output
    const dWk = matrixMultiply(transpose(x), dK); // (C, H)
    const dWq = matrixMultiply(transpose(x), dQ); // (C, H)
    const dWv = matrixMultiply(transpose(x), dV); // (C, H)

    // Gradient w.r.t. input: sum contributions from all three paths
    const dX = sum2d(
      sum2d(matrixMultiply(dQ, transpose(this.query.weights)), matrixMultiply(dK, transpose(this.key.weights))),
      matrixMultiply(dV, transpose(this.value.weights)),
    ); // (T, C)

    return { dX, dWk, dWq, dWv };
  }

  // x: (B, T, embeddingSize) -> (B, T, headSize)
  forward(x: Tensor3d): Tensor3d {
    const embeddingSize = x[0][0].length;
    const scale = Math.pow(embeddingSize, -0.5);

    return x.map((batch) => {
      const k = batch.map((token) => this.key.forward(token)); // (T, headSize)
      const q = batch.map((token) => this.query.forward(token)); // (T, headSize)
      const v = batch.map((token) => this.value.forward(token)); // (T, headSize)

      // wei = q @ k^T * scale -> (T, T)
      const wei: Tensor2d = matrixMultiply(q, transpose(k)).map((row) => row.map((w) => w * scale));

      // Causal mask: future positions get -Infinity so softmax zeroes them out (future tokens cannot influence the current token)
      const maskedWei: Tensor2d = wei.map((row, i) => row.map((w, j) => (j <= i ? w : -Infinity)));

      // Token affinities matrix (lower triangular matrix)
      const weightedWei = maskedWei.map(softmax); // (T, T)

      // out = wei @ v -> (T, headSize)
      return matrixMultiply(weightedWei, v);
    });
  }
}
