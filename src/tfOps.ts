import {
  sum2d,
  sum3d,
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
          (gradients['positionEmbedding'] as Tensor2d)[t][i] += dX[t][i];
        }
      }
    }

    return gradients;
  }
}

export class BigramLanguageModelWithFF implements Trainable {
  readonly tokenEmbeddingTable: Tensor2d;
  readonly positionEmbeddingTable: Tensor2d;
  readonly multiHeadAttention: MultiHeadAttention;
  readonly feedForward: FeedForward;
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
    this.feedForward = new FeedForward(numberEmbeddingDimensions);
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
    const feedForwardOut = this.feedForward.forward(attended); // (B, T, C)
    const logits = feedForwardOut.map((batch) => batch.map((token) => this.languageModelingHead.forward(token))); // (B, T, vocabSize)

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
      { name: 'ff1Weights', data: this.feedForward.linear1.weights },
      { name: 'ff1Bias', data: this.feedForward.linear1.bias },
      { name: 'ff2Weights', data: this.feedForward.linear2.weights },
      { name: 'ff2Bias', data: this.feedForward.linear2.bias },
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
    const feedForwardOut = this.feedForward.forward(attended); // (B, T, C)
    const logits = feedForwardOut.map((batch) => batch.map((token) => this.languageModelingHead.forward(token))); // (B, T, vocabSize)

    // Initialize gradient accumulators
    const gradients: { [paramName: string]: Tensor2d | Tensor1d } = {
      tokenEmbedding: this.tokenEmbeddingTable.map((row) => new Array<number>(row.length).fill(0)),
      positionEmbedding: this.positionEmbeddingTable.map((row) => new Array<number>(row.length).fill(0)),
      ff1Weights: this.feedForward.linear1.weights.map((row) => new Array<number>(row.length).fill(0)),
      ff1Bias: new Array<number>(this.feedForward.linear1.bias.length).fill(0),
      ff2Weights: this.feedForward.linear2.weights.map((row) => new Array<number>(row.length).fill(0)),
      ff2Bias: new Array<number>(this.feedForward.linear2.bias.length).fill(0),
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
        for (let i = 0; i < feedForwardOut[b][t].length; i++) {
          for (let j = 0; j < dLogits[t].length; j++) {
            (gradients['lmWeights'] as Tensor2d)[i][j] += feedForwardOut[b][t][i] * dLogits[t][j];
          }
        }
        for (let j = 0; j < dLogits[t].length; j++) {
          (gradients['lmBias'] as Tensor1d)[j] += dLogits[t][j];
        }
      }

      // d_feedForwardOut = dLogits @ Wlm^T
      const dFeedForwardOut = matrixMultiply(dLogits, transpose(this.languageModelingHead.weights));

      // Backward through feed forward
      const { dX: dAttended, dW1, dB1, dW2, dB2 } = this.feedForward.backward(attended[b], dFeedForwardOut);

      // Accumulate FF gradients
      for (let i = 0; i < dW1.length; i++) {
        for (let j = 0; j < dW1[i].length; j++) {
          (gradients['ff1Weights'] as Tensor2d)[i][j] += dW1[i][j];
        }
      }
      for (let j = 0; j < dB1.length; j++) {
        (gradients['ff1Bias'] as Tensor1d)[j] += dB1[j];
      }
      for (let i = 0; i < dW2.length; i++) {
        for (let j = 0; j < dW2[i].length; j++) {
          (gradients['ff2Weights'] as Tensor2d)[i][j] += dW2[i][j];
        }
      }
      for (let j = 0; j < dB2.length; j++) {
        (gradients['ff2Bias'] as Tensor1d)[j] += dB2[j];
      }

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
          (gradients['positionEmbedding'] as Tensor2d)[t][i] += dX[t][i];
        }
      }
    }

    return gradients;
  }
}

export class TransformerBlock {
  readonly ln1: LayerNorm;
  readonly multiHeadAttention: MultiHeadAttention;
  readonly ln2: LayerNorm;
  readonly feedForward: FeedForward;

  constructor(embeddingSize: number, numHeads: number) {
    this.ln1 = new LayerNorm(embeddingSize);
    this.multiHeadAttention = new MultiHeadAttention(embeddingSize, numHeads);
    this.ln2 = new LayerNorm(embeddingSize);
    this.feedForward = new FeedForward(embeddingSize);
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

  // x: (T, C), dOut: (T, C) -> gradients for all components
  backward(
    x: Tensor2d,
    dOut: Tensor2d,
  ): {
    dX: Tensor2d;
    ln1Grads: { dGamma: Tensor1d; dBeta: Tensor1d };
    attnGrads: Array<{ dWk: Tensor2d; dWq: Tensor2d; dWv: Tensor2d }>;
    ln2Grads: { dGamma: Tensor1d; dBeta: Tensor1d };
    ffGrads: { dW1: Tensor2d; dB1: Tensor1d; dW2: Tensor2d; dB2: Tensor1d };
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
}

export class LayerNorm {
  readonly gamma: Tensor1d; // learnable scale parameters
  readonly beta: Tensor1d;  // learnable shift parameters
  readonly eps: number;

  constructor(embeddingSize: number, eps = 1e-5) {
    this.eps = eps;
    this.gamma = new Array<number>(embeddingSize).fill(1.0); // initialize to 1
    this.beta = new Array<number>(embeddingSize).fill(0.0);  // initialize to 0
  }

  // x: (B, T, C) -> (B, T, C)
  forward(x: Tensor3d): Tensor3d {
    return x.map((batch) =>
      batch.map((token) => {
        // Calculate mean and variance for this token
        const mean = token.reduce((sum, val) => sum + val, 0) / token.length;
        const variance = token.reduce((sum, val) => sum + (val - mean) ** 2, 0) / token.length;
        const std = Math.sqrt(variance + this.eps);

        // Normalize and apply learnable parameters
        return token.map((val, i) => ((val - mean) / std) * this.gamma[i] + this.beta[i]);
      }),
    );
  }

  // x: (T, C), dOut: (T, C) -> gradients w.r.t. x, gamma, beta
  backward(x: Tensor2d, dOut: Tensor2d): { dX: Tensor2d; dGamma: Tensor1d; dBeta: Tensor1d } {
    const T = x.length;
    const C = x[0].length;

    // Recompute forward pass values needed for backward
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

        // Gradients w.r.t. gamma and beta
        dGamma[i] += normalized * dOut[t][i];
        dBeta[i] += dOut[t][i];
      }

      // Gradient w.r.t. input (more complex due to mean/variance dependencies)
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
}

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
  backward(x: Tensor2d, dOut: Tensor2d): { dX: Tensor2d; dW1: Tensor2d; dB1: Tensor1d; dW2: Tensor2d; dB2: Tensor1d } {
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

export class GPTModel implements Trainable {
  readonly tokenEmbeddingTable: Tensor2d;
  readonly positionEmbeddingTable: Tensor2d;
  readonly blocks: TransformerBlock[];
  readonly lnFinal: LayerNorm;
  readonly languageModelingHead: Linear;
  readonly contextSize: number;

  constructor(
    vocabSize: number,
    numberEmbeddingDimensions: number,
    contextSize: number,
    numHeads: number,
    numLayers: number,
  ) {
    this.contextSize = contextSize;
    this.tokenEmbeddingTable = Array.from({ length: vocabSize }, () =>
      Array.from({ length: numberEmbeddingDimensions }, () => random() * 0.01),
    );
    this.positionEmbeddingTable = Array.from({ length: contextSize }, () =>
      Array.from({ length: numberEmbeddingDimensions }, () => random() * 0.01),
    );

    // Create multiple transformer blocks
    this.blocks = Array.from({ length: numLayers }, () => new TransformerBlock(numberEmbeddingDimensions, numHeads));
    this.lnFinal = new LayerNorm(numberEmbeddingDimensions);
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
    let x = tokenEmbeddings.map((batch) => sum2d(batch, positionEmbeddings)); // (B, T, C)

    // Pass through all transformer blocks
    for (const block of this.blocks) {
      x = block.forward(x);
    }

    // Final layer norm and language modeling head
    const normalized = this.lnFinal.forward(x);
    const logits = normalized.map((batch) => batch.map((token) => this.languageModelingHead.forward(token))); // (B, T, vocabSize)

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
      { name: 'lnFinal_gamma', data: this.lnFinal.gamma },
      { name: 'lnFinal_beta', data: this.lnFinal.beta },
      { name: 'lmWeights', data: this.languageModelingHead.weights },
      { name: 'lmBias', data: this.languageModelingHead.bias },
    ];

    // Add parameters from all transformer blocks
    this.blocks.forEach((block, layerIdx) => {
      // LayerNorm 1
      params.push(
        { name: `layer${layerIdx}_ln1_gamma`, data: block.ln1.gamma },
        { name: `layer${layerIdx}_ln1_beta`, data: block.ln1.beta },
      );

      // Attention heads
      block.multiHeadAttention.heads.forEach((head, headIdx) => {
        params.push(
          { name: `layer${layerIdx}_head${headIdx}_key`, data: head.key.weights },
          { name: `layer${layerIdx}_head${headIdx}_query`, data: head.query.weights },
          { name: `layer${layerIdx}_head${headIdx}_value`, data: head.value.weights },
        );
      });

      // LayerNorm 2 & FeedForward
      params.push(
        { name: `layer${layerIdx}_ln2_gamma`, data: block.ln2.gamma },
        { name: `layer${layerIdx}_ln2_beta`, data: block.ln2.beta },
        { name: `layer${layerIdx}_ff1Weights`, data: block.feedForward.linear1.weights },
        { name: `layer${layerIdx}_ff1Bias`, data: block.feedForward.linear1.bias },
        { name: `layer${layerIdx}_ff2Weights`, data: block.feedForward.linear2.weights },
        { name: `layer${layerIdx}_ff2Bias`, data: block.feedForward.linear2.bias },
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
    let x = tokenEmbeddings.map((batch) => sum2d(batch, positionEmbeddings)); // (B, T, C)

    // Save intermediate activations for backward pass
    const activations: Tensor3d[] = [x];
    for (const block of this.blocks) {
      x = block.forward(x);
      activations.push(x);
    }

    const normalized = this.lnFinal.forward(x);
    const logits = normalized.map((batch) => batch.map((token) => this.languageModelingHead.forward(token))); // (B, T, vocabSize)

    // Initialize gradient accumulators
    const gradients: { [paramName: string]: Tensor2d | Tensor1d } = {
      tokenEmbedding: this.tokenEmbeddingTable.map((row) => new Array<number>(row.length).fill(0)),
      positionEmbedding: this.positionEmbeddingTable.map((row) => new Array<number>(row.length).fill(0)),
      lnFinal_gamma: new Array<number>(this.lnFinal.gamma.length).fill(0),
      lnFinal_beta: new Array<number>(this.lnFinal.beta.length).fill(0),
      lmWeights: this.languageModelingHead.weights.map((row) => new Array<number>(row.length).fill(0)),
      lmBias: new Array<number>(this.languageModelingHead.bias.length).fill(0),
    };

    // Initialize block gradients
    this.blocks.forEach((block, layerIdx) => {
      gradients[`layer${layerIdx}_ln1_gamma`] = new Array<number>(block.ln1.gamma.length).fill(0);
      gradients[`layer${layerIdx}_ln1_beta`] = new Array<number>(block.ln1.beta.length).fill(0);
      gradients[`layer${layerIdx}_ln2_gamma`] = new Array<number>(block.ln2.gamma.length).fill(0);
      gradients[`layer${layerIdx}_ln2_beta`] = new Array<number>(block.ln2.beta.length).fill(0);
      gradients[`layer${layerIdx}_ff1Weights`] = block.feedForward.linear1.weights.map((row) => new Array<number>(row.length).fill(0));
      gradients[`layer${layerIdx}_ff1Bias`] = new Array<number>(block.feedForward.linear1.bias.length).fill(0);
      gradients[`layer${layerIdx}_ff2Weights`] = block.feedForward.linear2.weights.map((row) => new Array<number>(row.length).fill(0));
      gradients[`layer${layerIdx}_ff2Bias`] = new Array<number>(block.feedForward.linear2.bias.length).fill(0);

      block.multiHeadAttention.heads.forEach((head, headIdx) => {
        gradients[`layer${layerIdx}_head${headIdx}_key`] = head.key.weights.map((row) => new Array<number>(row.length).fill(0));
        gradients[`layer${layerIdx}_head${headIdx}_query`] = head.query.weights.map((row) => new Array<number>(row.length).fill(0));
        gradients[`layer${layerIdx}_head${headIdx}_value`] = head.value.weights.map((row) => new Array<number>(row.length).fill(0));
      });
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
        for (let i = 0; i < normalized[b][t].length; i++) {
          for (let j = 0; j < dLogits[t].length; j++) {
            (gradients['lmWeights'] as Tensor2d)[i][j] += normalized[b][t][i] * dLogits[t][j];
          }
        }
        for (let j = 0; j < dLogits[t].length; j++) {
          (gradients['lmBias'] as Tensor1d)[j] += dLogits[t][j];
        }
      }

      // d_normalized = dLogits @ Wlm^T
      let dCurrent = matrixMultiply(dLogits, transpose(this.languageModelingHead.weights));

      // Backward through final LayerNorm
      const { dX: dFinalX, dGamma: dFinalGamma, dBeta: dFinalBeta } = this.lnFinal.backward(activations[activations.length - 1][b], dCurrent);
      
      for (let i = 0; i < dFinalGamma.length; i++) {
        (gradients['lnFinal_gamma'] as Tensor1d)[i] += dFinalGamma[i];
        (gradients['lnFinal_beta'] as Tensor1d)[i] += dFinalBeta[i];
      }

      dCurrent = dFinalX;

      // Backward through transformer blocks (in reverse order)
      for (let layerIdx = this.blocks.length - 1; layerIdx >= 0; layerIdx--) {
        const block = this.blocks[layerIdx];
        const blockInput = activations[layerIdx][b];

        const { dX, ln1Grads, attnGrads, ln2Grads, ffGrads } = block.backward(blockInput, dCurrent);

        // Accumulate gradients
        for (let i = 0; i < ln1Grads.dGamma.length; i++) {
          (gradients[`layer${layerIdx}_ln1_gamma`] as Tensor1d)[i] += ln1Grads.dGamma[i];
          (gradients[`layer${layerIdx}_ln1_beta`] as Tensor1d)[i] += ln1Grads.dBeta[i];
          (gradients[`layer${layerIdx}_ln2_gamma`] as Tensor1d)[i] += ln2Grads.dGamma[i];
          (gradients[`layer${layerIdx}_ln2_beta`] as Tensor1d)[i] += ln2Grads.dBeta[i];
        }

        // FF gradients
        for (let i = 0; i < ffGrads.dW1.length; i++) {
          for (let j = 0; j < ffGrads.dW1[i].length; j++) {
            (gradients[`layer${layerIdx}_ff1Weights`] as Tensor2d)[i][j] += ffGrads.dW1[i][j];
            (gradients[`layer${layerIdx}_ff2Weights`] as Tensor2d)[i][j] += ffGrads.dW2[i][j];
          }
        }
        for (let j = 0; j < ffGrads.dB1.length; j++) {
          (gradients[`layer${layerIdx}_ff1Bias`] as Tensor1d)[j] += ffGrads.dB1[j];
        }
        for (let j = 0; j < ffGrads.dB2.length; j++) {
          (gradients[`layer${layerIdx}_ff2Bias`] as Tensor1d)[j] += ffGrads.dB2[j];
        }

        // Attention gradients
        attnGrads.forEach((headGrad, headIdx) => {
          const { dWk, dWq, dWv } = headGrad;
          for (let i = 0; i < dWk.length; i++) {
            for (let j = 0; j < dWk[i].length; j++) {
              (gradients[`layer${layerIdx}_head${headIdx}_key`] as Tensor2d)[i][j] += dWk[i][j];
              (gradients[`layer${layerIdx}_head${headIdx}_query`] as Tensor2d)[i][j] += dWq[i][j];
              (gradients[`layer${layerIdx}_head${headIdx}_value`] as Tensor2d)[i][j] += dWv[i][j];
            }
          }
        });

        dCurrent = dX;
      }

      // Backward through embedding lookup
      for (let t = 0; t < T; t++) {
        const token = contextTokens[b][t];
        for (let i = 0; i < dCurrent[t].length; i++) {
          // Token embedding gradients (sparse update)
          (gradients['tokenEmbedding'] as Tensor2d)[token][i] += dCurrent[t][i];
          // Position embedding gradients (dense update) 
          (gradients['positionEmbedding'] as Tensor2d)[t][i] += dCurrent[t][i];
        }
      }
    }

    return gradients;
  }
}
