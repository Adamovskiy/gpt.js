import { softmax, type Tensor1d, type Tensor2d, type Tensor3d } from './tensorOps.js';
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

export class BigramLanguageModel {
  private readonly embedding: Tensor2d; // vocabSize x vocabSize

  constructor(vocabSize: number) {
    this.embedding = Array.from(Array(vocabSize), () => new Array(vocabSize).fill(0).map(() => random() * 0.01));
    this.m = this.embedding.map((row) => new Array(row.length).fill(0));
    this.v = this.embedding.map((row) => new Array(row.length).fill(0));
  }

  private m: Tensor2d;
  private v: Tensor2d;
  private t = 0;

  trainAdamW(contextTokens: Tensor2d, outputs: Tensor2d) {
    const learningRate = 1e-3;
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;
    const weightDecay = 0.01;

    this.t++;

    const { logits, loss } = this.forward(contextTokens, outputs);

    const countTokens = contextTokens.reduce((sum, batch) => sum + batch.length, 0);
    const scale = 1 / countTokens;

    // Collect gradients per token before applying to the embedding
    const gradEmbedding = this.embedding.map((tEmbedding) => new Array<number>(tEmbedding.length).fill(0));

    for (let batchIdx = 0; batchIdx < logits.length; batchIdx++) {
      const batchContextTokens = contextTokens[batchIdx];
      for (let tokenIdx = 0; tokenIdx < batchContextTokens.length; tokenIdx++) {
        const token = batchContextTokens[tokenIdx];
        const itsLogits = logits[batchIdx][tokenIdx];
        const probabilities = softmax(itsLogits);
        const targetToken = outputs[batchIdx][tokenIdx];

        const gradients = probabilities;
        gradients[targetToken] -= 1;

        const tokenGradEmbedding = gradEmbedding[token];
        for (let gradientIdx = 0; gradientIdx < tokenGradEmbedding.length; gradientIdx++) {
          tokenGradEmbedding[gradientIdx] += gradients[gradientIdx] * scale;
        }
      }
    }

    for (let token = 0; token < this.embedding.length; token++) {
      for (let i = 0; i < this.embedding[token].length; i++) {
        const g = gradEmbedding[token][i];

        // update moments
        this.m[token][i] = beta1 * this.m[token][i] + (1 - beta1) * g;
        this.v[token][i] = beta2 * this.v[token][i] + (1 - beta2) * g * g;

        // bias correction
        const mHat = this.m[token][i] / (1 - Math.pow(beta1, this.t));
        const vHat = this.v[token][i] / (1 - Math.pow(beta2, this.t));

        // update weights
        this.embedding[token][i] -= (learningRate * mHat) / (Math.sqrt(vHat) + eps);

        // weight decay (AdamW style)
        this.embedding[token][i] -= learningRate * weightDecay * this.embedding[token][i];
      }
    }

    return loss;
  }

  trainSGD(contextTokens: Tensor2d, outputs: Tensor2d) {
    const learningRate = 1e-3;
    const { logits, loss } = this.forward(contextTokens, outputs);

    const countTokens = contextTokens.reduce((sum, batch) => sum + batch.length, 0);
    const scale = 1 / countTokens;

    // Collect gradients per token before applying to the embedding
    const gradEmbedding = this.embedding.map((tEmbedding) => new Array<number>(tEmbedding.length).fill(0));

    for (let batchIdx = 0; batchIdx < logits.length; batchIdx++) {
      const batchContextTokens = contextTokens[batchIdx];
      for (let tokenIdx = 0; tokenIdx < batchContextTokens.length; tokenIdx++) {
        const token = batchContextTokens[tokenIdx];
        const itsLogits = logits[batchIdx][tokenIdx];
        const probabilities = softmax(itsLogits);
        const targetToken = outputs[batchIdx][tokenIdx];

        const gradients = probabilities;
        gradients[targetToken] -= 1;

        const tokenGradEmbedding = gradEmbedding[token];
        for (let gradientIdx = 0; gradientIdx < tokenGradEmbedding.length; gradientIdx++) {
          tokenGradEmbedding[gradientIdx] += gradients[gradientIdx] * scale;
        }
      }
    }

    for (let token = 0; token < this.embedding.length; token++) {
      for (let i = 0; i < this.embedding[token].length; i++) {
        this.embedding[token][i] -= learningRate * gradEmbedding[token][i];
      }
    }

    return loss;
  }

  forward(
    idx: Tensor2d, // (B, T)
    targets?: Tensor2d, // (B, T)
  ): {
    logits: Tensor3d; // (B, T, C)
    loss?: number;
  } {
    const logits = idx.map((batch) => batch.map((token) => this.embedding[token]));

    if (!targets) return { logits };
    const loss = crossEntropy(logits, targets);

    return { logits, loss };
  }

  generate(
    idx: Tensor2d, // (B, T, C)
    maxNewTokens: number,
  ) {
    for (let i = 0; i < maxNewTokens; i++) {
      const { logits } = this.forward(idx);

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
