import type { Tensor1d, Tensor2d, Tensor3d } from './tensorOps.js';

export function crossEntropy(logits: Tensor3d, targets: Tensor2d) {
  let sum = 0;
  let count = 0;

  for (let sampleIdx = 0; sampleIdx < logits.length; sampleIdx++) {
    for (let tokenIdx = 0; tokenIdx < logits[sampleIdx].length; tokenIdx++) {
      // Calculate loss for each token
      const tokenLogits = logits[sampleIdx][tokenIdx];
      const targetIndex = targets[sampleIdx][tokenIdx];
      // TODO do not calculate all probabilities, just one
      const probabilities = softmax(tokenLogits);
      sum += -Math.log(probabilities[targetIndex] + 1e-9); // Make sure it's never log(0)
      count++;
    }
  }

  return sum / count;
}

export class BigramLanguageModel {
  private readonly embedding: Tensor2d; // vocabSize x vocabSize

  constructor(vocabSize: number) {
    this.embedding = Array.from(Array(vocabSize), () => new Array(vocabSize).fill(0));

    // TODO remove
    for (let i = 0; i < vocabSize; i++) {
      this.embedding[i][i] = 1;
    }
  }

  forward(
    idx: Tensor2d, // (B, T, C)
    targets?: Tensor2d, // (B, T)
  ) {
    const logits = idx.map((batch) => batch.map((token) => this.embedding[token]));

    if (!targets) return { logits };
    const loss = crossEntropy(logits, targets);

    return { logits, loss };
  }
}

export function softmax(logits: Tensor1d) {
  const maxLogit = Math.max(...logits);
  const exp = logits.map((logit) => Math.exp(logit - maxLogit));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((exp) => exp / sum);
}
