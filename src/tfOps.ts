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
import type { Trainable, Parameter } from './types.js';

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

export class BigramLanguageModelMultiHeadAttention implements Trainable {
  readonly tokenEmbeddingTable: Tensor2d;
  readonly positionEmbeddingTable: Tensor2d;
  readonly multiHeadAttention: MultiHeadAttention;
  readonly languageModelingHead: Linear;
  readonly contextSize: number;

  constructor(vocabSize: number, numberEmbeddingDimensions: number, contextSize: number, numHeads: number) {
    this.contextSize = contextSize;
    this.tokenEmbeddingTable = Array.from({ length: vocabSize }, () =>
      Array.from({ length: numberEmbeddingDimensions }, () => random() * 0.01),
    );
    this.positionEmbeddingTable = Array.from({ length: contextSize }, () =>
      Array.from({ length: numberEmbeddingDimensions }, () => random() * 0.01),
    );

    this.multiHeadAttention = new MultiHeadAttention(numberEmbeddingDimensions, numHeads);
    this.languageModelingHead = new Linear(numberEmbeddingDimensions, vocabSize);
  }

  forward(
    idx: Tensor2d, // (B, T)
    targets?: Tensor2d, // (B, T)
  ): {
    logits: Tensor3d; // (B, T, C)
    loss?: number;
  } {
    const tokenEmbeddings = idx.map((batch) => batch.map((token) => this.tokenEmbeddingTable[token])); // (B, T, C)
    const positionEmbeddings = idx[0].map((_, i) => this.positionEmbeddingTable[i]); // (T, C)
    const embeddingsSum = tokenEmbeddings.map((batch) => sum2d(batch, positionEmbeddings)); // (B, T, C)
    const attended = this.multiHeadAttention.forward(embeddingsSum); // (B, T, C)
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
      const idxCond = idx.map((batch) => batch.slice(-this.contextSize)); // crop to contextSize
      const { logits } = this.forward(idxCond);

      const lastTokenLogits = logits.map((batch) => batch[batch.length - 1]); // (B, C)
      const probs = softmaxBatched(lastTokenLogits); // (B, C)
      const idxNext = sampleMultinomial(probs);
      concatBatched(idx, idxNext);
    }

    return idx;
  }

  getParameters(): Parameter[] {
    const params: Parameter[] = [
      { name: 'tokenEmbedding', data: this.tokenEmbeddingTable },
      { name: 'positionEmbedding', data: this.positionEmbeddingTable },
      { name: 'lmWeights', data: this.languageModelingHead.weights },
      { name: 'lmBias', data: this.languageModelingHead.bias },
    ];

    // Add parameters from each attention head
    this.multiHeadAttention.heads.forEach((head, headIdx) => {
      params.push(
        { name: `head${headIdx}_key`, data: head.key.weights },
        { name: `head${headIdx}_query`, data: head.query.weights },
        { name: `head${headIdx}_value`, data: head.value.weights },
      );
    });

    return params;
  }

  computeGradients(contextTokens: Tensor2d, targets: Tensor2d): { [paramName: string]: Tensor2d | Tensor1d } {
    const B = contextTokens.length;
    const T = contextTokens[0].length;
    const scale = 1 / (B * T);

    // Forward pass (save intermediates for backward)
    const tokenEmbeddings = contextTokens.map((batch) => batch.map((token) => this.tokenEmbeddingTable[token])); // (B, T, C)
    const positionEmbeddings = contextTokens[0].map((_, i) => this.positionEmbeddingTable[i]); // (T, C)
    const embeddingsSum = tokenEmbeddings.map((batch) => sum2d(batch, positionEmbeddings)); // (B, T, C)
    const attended = this.multiHeadAttention.forward(embeddingsSum); // (B, T, C)
    const logits = attended.map((batch) => batch.map((token) => this.languageModelingHead.forward(token))); // (B, T, vocabSize)

    // Initialize gradient accumulators
    const gradients: { [paramName: string]: Tensor2d | Tensor1d } = {
      tokenEmbedding: this.tokenEmbeddingTable.map((row) => new Array<number>(row.length).fill(0)),
      positionEmbedding: this.positionEmbeddingTable.map((row) => new Array<number>(row.length).fill(0)),
      lmWeights: this.languageModelingHead.weights.map((row) => new Array<number>(row.length).fill(0)),
      lmBias: new Array<number>(this.languageModelingHead.bias.length).fill(0),
    };

    // Initialize head gradients
    this.multiHeadAttention.heads.forEach((head, headIdx) => {
      gradients[`head${headIdx}_key`] = head.key.weights.map((row) => new Array<number>(row.length).fill(0));
      gradients[`head${headIdx}_query`] = head.query.weights.map((row) => new Array<number>(row.length).fill(0));
      gradients[`head${headIdx}_value`] = head.value.weights.map((row) => new Array<number>(row.length).fill(0));
    });

    for (let b = 0; b < B; b++) {
      // d_logits = (softmax(logits) - one_hot(target)) * scale
      const dLogits = logits[b].map((tokenLogits, t) => {
        const probs = softmax(tokenLogits);
        probs[targets[b][t]] -= 1;
        return probs.map((v) => v * scale);
      }); // (T, vocabSize)

      // Backward through LM head
      for (let t = 0; t < T; t++) {
        for (let i = 0; i < attended[b][t].length; i++) {
          for (let j = 0; j < dLogits[t].length; j++) {
            (gradients['lmWeights'] as Tensor2d)[i][j] += attended[b][t][i] * dLogits[t][j];
          }
        }
        for (let j = 0; j < dLogits[t].length; j++) {
          (gradients['lmBias'] as Tensor1d)[j] += dLogits[t][j];
        }
      }

      // d_attended = dLogits @ Wlm^T
      const dAttended = matrixMultiply(dLogits, transpose(this.languageModelingHead.weights));

      // Backward through multi-head attention
      const { dX, headGrads } = this.multiHeadAttention.backward(embeddingsSum[b], dAttended);

      // Accumulate head gradients
      headGrads.forEach((headGrad, headIdx) => {
        const { dWk, dWq, dWv } = headGrad;
        for (let i = 0; i < dWk.length; i++) {
          for (let j = 0; j < dWk[i].length; j++) {
            (gradients[`head${headIdx}_key`] as Tensor2d)[i][j] += dWk[i][j];
            (gradients[`head${headIdx}_query`] as Tensor2d)[i][j] += dWq[i][j];
            (gradients[`head${headIdx}_value`] as Tensor2d)[i][j] += dWv[i][j];
          }
        }
      });

      // Backward through embedding lookup
      for (let t = 0; t < T; t++) {
        const token = contextTokens[b][t];
        for (let i = 0; i < dX[t].length; i++) {
          (gradients['tokenEmbedding'] as Tensor2d)[token][i] += dX[t][i];
        }
      }
    }

    return gradients;
  }
}

export class MultiHeadAttention {
  readonly heads: Head[];
  readonly numHeads: number;
  readonly headSize: number;

  constructor(embeddingSize: number, numHeads: number) {
    this.numHeads = numHeads;
    this.headSize = Math.floor(embeddingSize / numHeads);
    this.heads = Array.from({ length: numHeads }, () => new Head(embeddingSize, this.headSize));
  }

  // x: (B, T, C) -> (B, T, C)
  forward(x: Tensor3d): Tensor3d {
    const headOutputs = this.heads.map((head) => head.forward(x)); // Array of (B, T, headSize)

    // Concatenate along the last dimension: (B, T, numHeads * headSize)
    return x.map((_, batchIdx) =>
      x[batchIdx].map((_, tokenIdx) => headOutputs.flatMap((headOutput) => headOutput[batchIdx][tokenIdx])),
    );
  }

  // x: (T, C), dOut: (T, C) -> gradients for all heads
  backward(
    x: Tensor2d,
    dOut: Tensor2d,
  ): { dX: Tensor2d; headGrads: Array<{ dWk: Tensor2d; dWq: Tensor2d; dWv: Tensor2d }> } {
    // Split dOut back into per-head gradients: (T, numHeads * headSize) -> numHeads × (T, headSize)
    const dOutPerHead = this.heads.map((_, headIdx) =>
      dOut.map((tokenGrad) => tokenGrad.slice(headIdx * this.headSize, (headIdx + 1) * this.headSize)),
    );

    // Backward through each head
    const headResults = this.heads.map((head, headIdx) => head.backward(x, dOutPerHead[headIdx]));

    // Sum dX contributions from all heads
    const dX = headResults.reduce(
      (acc, { dX: headDX }) => acc.map((row, i) => row.map((val, j) => val + headDX[i][j])),
      x.map((row) => new Array<number>(row.length).fill(0)),
    );

    const headGrads = headResults.map(({ dWk, dWq, dWv }) => ({ dWk, dWq, dWv }));

    return { dX, headGrads };
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
