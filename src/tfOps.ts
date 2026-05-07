import {
  sum2d,
  softmax,
  softmaxBackward,
  type Tensor1d,
  type Tensor2d,
  type Tensor3d,
  matrixMultiply,
  sum1d,
  transpose,
} from './tensorOps.js';
import { random } from './random.js';

export function crossEntropy(logits: Tensor3d, targets: Tensor2d) {
  let sum = 0;
  let count = 0;

  for (let batchIdx = 0; batchIdx < logits.length; batchIdx++) {
    for (let tokenIdx = 0; tokenIdx < logits[batchIdx].length; tokenIdx++) {
      // Calculate loss for each token
      const tokenLogits = logits[batchIdx][tokenIdx];
      const targetIndex = targets[batchIdx][tokenIdx];
      // TODO do not calculate all probabilities, just one
      const probabilities = softmax(tokenLogits);
      sum += -Math.log(probabilities[targetIndex] + 1e-9); // Make sure it's never log(0)
      count++;
    }
  }

  return sum / count;
}

export class Linear {
  readonly weights: number[][];
  readonly bias: number[];

  constructor(inputSize: number, outputSize: number, useBias = true) {
    this.weights = Array.from({ length: inputSize }, () => Array.from({ length: outputSize }, () => random() * 0.01));
    this.bias = useBias
      ? Array.from({ length: outputSize }, () => random() * 0.01)
      : new Array<number>(outputSize).fill(0);
  }

  // (inputSize) -> (outputSize)
  forward(input: Tensor1d): Tensor1d {
    return sum1d(matrixMultiply([input], this.weights)[0], this.bias);
  }
}

export class BigramLanguageModel {
  readonly tokenEmbeddingTable: Tensor2d; // vocabSize x numberEmbeddingDimensions
  readonly positionEmbeddingTable: Tensor2d; // blockSize x numberEmbeddingDimensions
  readonly languageModelingHead: Linear; // Transforms embeddings to logits
  readonly contextSize: number;

  constructor(vocabSize: number, numberEmbeddingDimensions: number, contextSize: number) {
    this.contextSize = contextSize;
    this.tokenEmbeddingTable = Array.from({ length: vocabSize }, () =>
      Array.from({ length: numberEmbeddingDimensions }, () => random() * 0.01),
    );
    this.positionEmbeddingTable = Array.from({ length: contextSize }, () =>
      Array.from({ length: numberEmbeddingDimensions }, () => random() * 0.01),
    );

    this.languageModelingHead = new Linear(numberEmbeddingDimensions, vocabSize);
  }

  forward(
    idx: Tensor2d, // (B, T)
    targets?: Tensor2d, // (B, T)
  ): {
    logits: Tensor3d; // (B, T, C)
    loss?: number;
  } {
    const tokenEmbeddings = idx.map((batch) => batch.map((token) => this.tokenEmbeddingTable[token])); // (B,T, numberEmbeddingDimensions)
    const positionEmbeddings = idx[0].map((_, i) => this.positionEmbeddingTable[i]); // (T, numberEmbeddingDimensions)
    const embeddingsSum = tokenEmbeddings.map((batch) => sum2d(batch, positionEmbeddings)); // (T, numberEmbeddingDimensions)
    const logits = embeddingsSum.map((batch) => batch.map((token) => this.languageModelingHead.forward(token))); // (B,T, vocabSize)

    if (!targets) return { logits };
    const loss = crossEntropy(logits, targets);

    return { logits, loss };
  }

  generate(
    idx: Tensor2d, // (B, T, C)
    maxNewTokens: number,
  ) {
    for (let i = 0; i < maxNewTokens; i++) {
      const idxCond = idx.map((batch) => batch.slice(-this.contextSize)); // crop to blockSize
      const { logits } = this.forward(idxCond);

      const lastTokenLogits = logits.map((batch) => batch[batch.length - 1]); // (B, C)
      const probs = softmaxBatched(lastTokenLogits); // (B, C)
      const idxNext = sampleMultinomial(probs);
      concatBatched(idx, idxNext);
    }

    return idx;
  }
}

export class BigramLanguageModelSingleHeadAttention {
  readonly tokenEmbeddingTable: Tensor2d; // vocabSize x numberEmbeddingDimensions
  readonly positionEmbeddingTable: Tensor2d; // blockSize x numberEmbeddingDimensions
  readonly selfAttention: Head; // single self-attention head
  readonly languageModelingHead: Linear; // Transforms attended embeddings to logits
  readonly contextSize: number;

  constructor(vocabSize: number, numberEmbeddingDimensions: number, contextSize: number) {
    this.contextSize = contextSize;
    this.tokenEmbeddingTable = Array.from({ length: vocabSize }, () =>
      Array.from({ length: numberEmbeddingDimensions }, () => random() * 0.01),
    );
    this.positionEmbeddingTable = Array.from({ length: contextSize }, () =>
      Array.from({ length: numberEmbeddingDimensions }, () => random() * 0.01),
    );

    this.selfAttention = new Head(numberEmbeddingDimensions, numberEmbeddingDimensions);
    this.languageModelingHead = new Linear(numberEmbeddingDimensions, vocabSize);
  }

  forward(
    idx: Tensor2d, // (B, T)
    targets?: Tensor2d, // (B, T)
  ): {
    logits: Tensor3d; // (B, T, C)
    loss?: number;
  } {
    const tokenEmbeddings = idx.map((batch) => batch.map((token) => this.tokenEmbeddingTable[token])); // (B, T, numberEmbeddingDimensions)
    const positionEmbeddings = idx[0].map((_, i) => this.positionEmbeddingTable[i]); // (T, numberEmbeddingDimensions)
    const embeddingsSum = tokenEmbeddings.map((batch) => sum2d(batch, positionEmbeddings)); // (B, T, numberEmbeddingDimensions)
    const attended = this.selfAttention.forward(embeddingsSum); // (B, T, numberEmbeddingDimensions)
    const logits = attended.map((batch) => batch.map((token) => this.languageModelingHead.forward(token))); // (B, T, vocabSize)

    if (!targets) return { logits };
    const loss = crossEntropy(logits, targets);

    return { logits, loss };
  }

  generate(
    idx: Tensor2d, // (B, T, C)
    maxNewTokens: number,
  ) {
    for (let i = 0; i < maxNewTokens; i++) {
      const idxCond = idx.map((batch) => batch.slice(-this.contextSize)); // crop to blockSize
      const { logits } = this.forward(idxCond);

      const lastTokenLogits = logits.map((batch) => batch[batch.length - 1]); // (B, C)
      const probs = softmaxBatched(lastTokenLogits); // (B, C)
      const idxNext = sampleMultinomial(probs);
      concatBatched(idx, idxNext);
    }

    return idx;
  }
}

export class Head {
  readonly key: Linear; // projects embedding -> headSize (no bias, as in nanoGPT)
  readonly query: Linear;
  readonly value: Linear;

  constructor(
    embeddingSize: number,
    readonly headSize: number,
  ) {
    this.key = new Linear(embeddingSize, headSize, false);
    this.query = new Linear(embeddingSize, headSize, false);
    this.value = new Linear(embeddingSize, headSize, false);
  }

  // x: (T, C), dOut: (T, headSize) -> gradients w.r.t. x and weight matrices
  backward(x: Tensor2d, dOut: Tensor2d): { dX: Tensor2d; dWk: Tensor2d; dWq: Tensor2d; dWv: Tensor2d } {
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

function sampleMultinomial(
  batches: Tensor2d, // (B, C)
) {
  return batches.map((probabilities) => {
    let sum = 0;
    const rnd = random();
    for (let idx = 0; idx < probabilities.length; idx++) {
      sum += probabilities[idx];
      if (sum > rnd) {
        return idx;
      }
    }
    return probabilities.length - 1; // Fallback
  });
}

function concatBatched(idx: Tensor2d, idxNext: Tensor1d) {
  for (let i = 0; i < idxNext.length; i++) {
    idx[i].push(idxNext[i]);
  }
}

export function softmaxBatched(batches: Tensor2d) {
  return batches.map((batch) => softmax(batch));
}
