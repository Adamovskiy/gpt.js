import { sum2d, softmax, type Tensor1d, type Tensor2d, type Tensor3d, matrixMultiply, sum1d } from './tensorOps.js';
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

  constructor(inputSize: number, outputSize: number) {
    this.weights = Array.from({ length: inputSize }, () => Array.from({ length: outputSize }, () => random() * 0.01));
    this.bias = Array.from({ length: outputSize }, () => random() * 0.01);
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
