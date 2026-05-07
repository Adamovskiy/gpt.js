import { softmax, transpose, matrixMultiply, type Tensor1d, type Tensor2d, type Tensor3d } from './tensorOps.js';
import type { Linear } from './tfOps.js';
import type { BigramLanguageModelSingleHeadAttention } from './tfOps.js';

export interface Model {
  forward(contextTokens: Tensor2d, outputs: Tensor2d): { logits: Tensor3d; loss?: number };
  tokenEmbeddingTable: Tensor2d;
  languageModelingHead: Linear;
}

export interface Optimizer {
  train(contextTokens: Tensor2d, outputs: Tensor2d): number;
}

export class SDGOptimizer implements Optimizer {
  constructor(
    private readonly model: Model,
    private readonly learningRate: number,
  ) {}

  train(contextTokens: Tensor2d, outputs: Tensor2d): number {
    const { logits, loss } = this.model.forward(contextTokens, outputs);

    const countTokens = contextTokens.reduce((sum, batch) => sum + batch.length, 0);
    const scale = 1 / countTokens;

    // Collect gradients per token before applying to the parameters
    const gradEmbedding = this.model.tokenEmbeddingTable.map((row) => new Array<number>(row.length).fill(0));
    const gradWeights = this.model.languageModelingHead.weights.map((row) => new Array<number>(row.length).fill(0));
    const gradBias = new Array<number>(this.model.languageModelingHead.bias.length).fill(0);

    for (let batchIdx = 0; batchIdx < logits.length; batchIdx++) {
      const batchContextTokens = contextTokens[batchIdx];

      for (let tokenIdx = 0; tokenIdx < batchContextTokens.length; tokenIdx++) {
        const token = batchContextTokens[tokenIdx];

        const embedding = this.model.tokenEmbeddingTable[token];
        const weights = this.model.languageModelingHead.weights;

        const itsLogits = logits[batchIdx][tokenIdx];
        const probabilities = softmax(itsLogits);

        const targetToken = outputs[batchIdx][tokenIdx];

        const gradients = probabilities;
        gradients[targetToken] -= 1;

        // --- grad W ---
        for (let i = 0; i < embedding.length; i++) {
          for (let j = 0; j < gradients.length; j++) {
            gradWeights[i][j] += embedding[i] * gradients[j] * scale;
          }
        }

        // --- grad bias ---
        for (let j = 0; j < gradients.length; j++) {
          gradBias[j] += gradients[j] * scale;
        }

        // --- grad embedding ---
        const gradEmbRow = gradEmbedding[token];

        for (let i = 0; i < embedding.length; i++) {
          let sum = 0;
          for (let j = 0; j < gradients.length; j++) {
            sum += gradients[j] * weights[i][j];
          }
          gradEmbRow[i] += sum * scale;
        }
      }
    }

    // --- update embedding ---
    for (let token = 0; token < this.model.tokenEmbeddingTable.length; token++) {
      for (let i = 0; i < this.model.tokenEmbeddingTable[token].length; i++) {
        this.model.tokenEmbeddingTable[token][i] -= this.learningRate * gradEmbedding[token][i];
      }
    }

    // --- update weights ---
    for (let i = 0; i < this.model.languageModelingHead.weights.length; i++) {
      for (let j = 0; j < this.model.languageModelingHead.weights[i].length; j++) {
        this.model.languageModelingHead.weights[i][j] -= this.learningRate * gradWeights[i][j];
      }
    }

    // --- update bias ---
    for (let j = 0; j < this.model.languageModelingHead.bias.length; j++) {
      this.model.languageModelingHead.bias[j] -= this.learningRate * gradBias[j];
    }

    return loss || 0;
  }
}

export class AdamWOptimizer implements Optimizer {
  private embeddingMomentum: Tensor2d;
  private embeddingVelocity: Tensor2d;

  private weightsMomentum: Tensor2d;
  private weightsVelocity: Tensor2d;

  private biasMomentum: Tensor1d;
  private biasVelocity: Tensor1d;

  private stepCount = 0;

  constructor(
    private readonly model: Model,
    private readonly learningRate: number,
    private readonly beta1: number,
    private readonly beta2: number,
    private readonly eps: number,
    private readonly weightDecay: number,
  ) {
    this.embeddingMomentum = model.tokenEmbeddingTable.map((row) => new Array<number>(row.length).fill(0));
    this.embeddingVelocity = model.tokenEmbeddingTable.map((row) => new Array<number>(row.length).fill(0));
    this.weightsMomentum = model.languageModelingHead.weights.map((row) => new Array<number>(row.length).fill(0));
    this.weightsVelocity = model.languageModelingHead.weights.map((row) => new Array<number>(row.length).fill(0));
    this.biasMomentum = new Array<number>(model.languageModelingHead.bias.length).fill(0);
    this.biasVelocity = new Array<number>(model.languageModelingHead.bias.length).fill(0);
  }

  train(contextTokens: Tensor2d, outputs: Tensor2d): number {
    this.stepCount++;
    const { logits, loss } = this.model.forward(contextTokens, outputs);

    const countTokens = contextTokens.reduce((sum, batch) => sum + batch.length, 0);
    const scale = 1 / countTokens;

    const gradEmbedding = this.model.tokenEmbeddingTable.map((row) => new Array<number>(row.length).fill(0));
    const gradWeights = this.model.languageModelingHead.weights.map((row) => new Array<number>(row.length).fill(0));
    const gradBias = new Array<number>(this.model.languageModelingHead.bias.length).fill(0);

    for (let batchIdx = 0; batchIdx < logits.length; batchIdx++) {
      const batchContextTokens = contextTokens[batchIdx];

      for (let tokenIdx = 0; tokenIdx < batchContextTokens.length; tokenIdx++) {
        const token = batchContextTokens[tokenIdx];

        const embedding = this.model.tokenEmbeddingTable[token];
        const weights = this.model.languageModelingHead.weights;

        const itsLogits = logits[batchIdx][tokenIdx];
        const probabilities = softmax(itsLogits);

        const targetToken = outputs[batchIdx][tokenIdx];

        const gradients = probabilities;
        gradients[targetToken] -= 1;

        // --- grad W ---
        for (let i = 0; i < embedding.length; i++) {
          for (let j = 0; j < gradients.length; j++) {
            gradWeights[i][j] += embedding[i] * gradients[j] * scale;
          }
        }

        // --- grad bias ---
        for (let j = 0; j < gradients.length; j++) {
          gradBias[j] += gradients[j] * scale;
        }

        // --- grad embedding ---
        const gradEmbRow = gradEmbedding[token];

        for (let i = 0; i < embedding.length; i++) {
          let sum = 0;
          for (let j = 0; j < gradients.length; j++) {
            sum += gradients[j] * weights[i][j];
          }
          gradEmbRow[i] += sum * scale;
        }
      }
    }

    const { beta1, beta2, eps, learningRate, weightDecay } = this;
    const biasCorrection1 = 1 - Math.pow(beta1, this.stepCount);
    const biasCorrection2 = 1 - Math.pow(beta2, this.stepCount);

    // --- update embedding ---
    for (let token = 0; token < this.model.tokenEmbeddingTable.length; token++) {
      for (let i = 0; i < this.model.tokenEmbeddingTable[token].length; i++) {
        const g = gradEmbedding[token][i];
        this.embeddingMomentum[token][i] = beta1 * this.embeddingMomentum[token][i] + (1 - beta1) * g;
        this.embeddingVelocity[token][i] = beta2 * this.embeddingVelocity[token][i] + (1 - beta2) * g * g;
        const mHat = this.embeddingMomentum[token][i] / biasCorrection1;
        const vHat = this.embeddingVelocity[token][i] / biasCorrection2;
        this.model.tokenEmbeddingTable[token][i] *= 1 - learningRate * weightDecay;
        this.model.tokenEmbeddingTable[token][i] -= learningRate * (mHat / (Math.sqrt(vHat) + eps));
      }
    }

    // --- update weights ---
    for (let i = 0; i < this.model.languageModelingHead.weights.length; i++) {
      for (let j = 0; j < this.model.languageModelingHead.weights[i].length; j++) {
        const g = gradWeights[i][j];
        this.weightsMomentum[i][j] = beta1 * this.weightsMomentum[i][j] + (1 - beta1) * g;
        this.weightsVelocity[i][j] = beta2 * this.weightsVelocity[i][j] + (1 - beta2) * g * g;
        const mHat = this.weightsMomentum[i][j] / biasCorrection1;
        const vHat = this.weightsVelocity[i][j] / biasCorrection2;
        this.model.languageModelingHead.weights[i][j] *= 1 - learningRate * weightDecay;
        this.model.languageModelingHead.weights[i][j] -= learningRate * (mHat / (Math.sqrt(vHat) + eps));
      }
    }

    // --- update bias ---
    for (let j = 0; j < this.model.languageModelingHead.bias.length; j++) {
      const g = gradBias[j];
      this.biasMomentum[j] = beta1 * this.biasMomentum[j] + (1 - beta1) * g;
      this.biasVelocity[j] = beta2 * this.biasVelocity[j] + (1 - beta2) * g * g;
      const mHat = this.biasMomentum[j] / biasCorrection1;
      const vHat = this.biasVelocity[j] / biasCorrection2;
      this.model.languageModelingHead.bias[j] *= 1 - learningRate * weightDecay;
      this.model.languageModelingHead.bias[j] -= learningRate * (mHat / (Math.sqrt(vHat) + eps));
    }

    return loss || 0;
  }
}

export class AdamWOptimizerSingleHeadAttention implements Optimizer {
  private embeddingMomentum: Tensor2d;
  private embeddingVelocity: Tensor2d;

  private wkMomentum: Tensor2d;
  private wkVelocity: Tensor2d;
  private wqMomentum: Tensor2d;
  private wqVelocity: Tensor2d;
  private wvMomentum: Tensor2d;
  private wvVelocity: Tensor2d;

  private weightsMomentum: Tensor2d;
  private weightsVelocity: Tensor2d;
  private biasMomentum: Tensor1d;
  private biasVelocity: Tensor1d;

  private stepCount = 0;

  constructor(
    private readonly model: BigramLanguageModelSingleHeadAttention,
    private readonly learningRate: number,
    private readonly beta1: number,
    private readonly beta2: number,
    private readonly eps: number,
    private readonly weightDecay: number,
  ) {
    const { tokenEmbeddingTable, languageModelingHead, selfAttention } = model;

    this.embeddingMomentum = tokenEmbeddingTable.map((row) => new Array<number>(row.length).fill(0));
    this.embeddingVelocity = tokenEmbeddingTable.map((row) => new Array<number>(row.length).fill(0));

    this.wkMomentum = selfAttention.key.weights.map((row) => new Array<number>(row.length).fill(0));
    this.wkVelocity = selfAttention.key.weights.map((row) => new Array<number>(row.length).fill(0));
    this.wqMomentum = selfAttention.query.weights.map((row) => new Array<number>(row.length).fill(0));
    this.wqVelocity = selfAttention.query.weights.map((row) => new Array<number>(row.length).fill(0));
    this.wvMomentum = selfAttention.value.weights.map((row) => new Array<number>(row.length).fill(0));
    this.wvVelocity = selfAttention.value.weights.map((row) => new Array<number>(row.length).fill(0));

    this.weightsMomentum = languageModelingHead.weights.map((row) => new Array<number>(row.length).fill(0));
    this.weightsVelocity = languageModelingHead.weights.map((row) => new Array<number>(row.length).fill(0));
    this.biasMomentum = new Array<number>(languageModelingHead.bias.length).fill(0);
    this.biasVelocity = new Array<number>(languageModelingHead.bias.length).fill(0);
  }

  train(contextTokens: Tensor2d, targets: Tensor2d): number {
    this.stepCount++;

    const { model } = this;
    const { tokenEmbeddingTable, positionEmbeddingTable, selfAttention, languageModelingHead } = model;

    const B = contextTokens.length;
    const T = contextTokens[0].length;
    const scale = 1 / (B * T);

    // Forward pass (save intermediates needed for backward)
    const tokenEmbeddings = contextTokens.map((batch) => batch.map((token) => tokenEmbeddingTable[token])); // (B, T, C)
    const positionEmbeddings = contextTokens[0].map((_, i) => positionEmbeddingTable[i]); // (T, C)
    const embeddingsSum = tokenEmbeddings.map((batch) =>
      batch.map((token, t) => token.map((v, c) => v + positionEmbeddings[t][c])),
    ); // (B, T, C)
    const attended = selfAttention.forward(embeddingsSum); // (B, T, C)
    const logits = attended.map((batch) => batch.map((token) => languageModelingHead.forward(token))); // (B, T, vocabSize)

    // Loss
    let lossSum = 0;
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < T; t++) {
        const probs = softmax(logits[b][t]);
        lossSum += -Math.log(probs[targets[b][t]] + 1e-9);
      }
    }
    const loss = lossSum / (B * T);

    // Gradient accumulators
    const gradEmb = tokenEmbeddingTable.map((row) => new Array<number>(row.length).fill(0));
    const gradWk = selfAttention.key.weights.map((row) => new Array<number>(row.length).fill(0));
    const gradWq = selfAttention.query.weights.map((row) => new Array<number>(row.length).fill(0));
    const gradWv = selfAttention.value.weights.map((row) => new Array<number>(row.length).fill(0));
    const gradWlm = languageModelingHead.weights.map((row) => new Array<number>(row.length).fill(0));
    const gradBlm = new Array<number>(languageModelingHead.bias.length).fill(0);

    for (let b = 0; b < B; b++) {
      // d_logits = (softmax(logits) - one_hot(target)) * scale
      const dLogits = logits[b].map((tokenLogits, t) => {
        const probs = softmax(tokenLogits);
        probs[targets[b][t]] -= 1;
        return probs.map((v) => v * scale);
      }); // (T, vocabSize)

      // Backward through LM head: logits = attended @ Wlm + blm
      for (let t = 0; t < T; t++) {
        for (let i = 0; i < attended[b][t].length; i++) {
          for (let j = 0; j < dLogits[t].length; j++) {
            gradWlm[i][j] += attended[b][t][i] * dLogits[t][j];
          }
        }
        for (let j = 0; j < dLogits[t].length; j++) {
          gradBlm[j] += dLogits[t][j];
        }
      }

      // d_attended = dLogits @ Wlm^T  (T, C)
      const dAttended = matrixMultiply(dLogits, transpose(languageModelingHead.weights));

      // Backward through attention head
      const { dX, dWk, dWq, dWv } = selfAttention.backward(embeddingsSum[b], dAttended);

      // Accumulate attention weight gradients
      for (let i = 0; i < gradWk.length; i++) {
        for (let j = 0; j < gradWk[i].length; j++) {
          gradWk[i][j] += dWk[i][j];
          gradWq[i][j] += dWq[i][j];
          gradWv[i][j] += dWv[i][j];
        }
      }

      // Backward through token embedding lookup
      for (let t = 0; t < T; t++) {
        const token = contextTokens[b][t];
        for (let i = 0; i < dX[t].length; i++) {
          gradEmb[token][i] += dX[t][i];
        }
      }
    }

    // AdamW update helper for 2D params
    const { beta1, beta2, eps, learningRate, weightDecay } = this;
    const bc1 = 1 - Math.pow(beta1, this.stepCount);
    const bc2 = 1 - Math.pow(beta2, this.stepCount);

    const adamwUpdate = (param: number[][], grad: number[][], momentum: number[][], velocity: number[][]) => {
      for (let i = 0; i < param.length; i++) {
        for (let j = 0; j < param[i].length; j++) {
          const g = grad[i][j];
          momentum[i][j] = beta1 * momentum[i][j] + (1 - beta1) * g;
          velocity[i][j] = beta2 * velocity[i][j] + (1 - beta2) * g * g;
          const mHat = momentum[i][j] / bc1;
          const vHat = velocity[i][j] / bc2;
          param[i][j] *= 1 - learningRate * weightDecay;
          param[i][j] -= learningRate * (mHat / (Math.sqrt(vHat) + eps));
        }
      }
    };

    adamwUpdate(tokenEmbeddingTable, gradEmb, this.embeddingMomentum, this.embeddingVelocity);
    adamwUpdate(selfAttention.key.weights, gradWk, this.wkMomentum, this.wkVelocity);
    adamwUpdate(selfAttention.query.weights, gradWq, this.wqMomentum, this.wqVelocity);
    adamwUpdate(selfAttention.value.weights, gradWv, this.wvMomentum, this.wvVelocity);
    adamwUpdate(languageModelingHead.weights, gradWlm, this.weightsMomentum, this.weightsVelocity);

    // Update LM head bias
    for (let j = 0; j < languageModelingHead.bias.length; j++) {
      const g = gradBlm[j];
      this.biasMomentum[j] = beta1 * this.biasMomentum[j] + (1 - beta1) * g;
      this.biasVelocity[j] = beta2 * this.biasVelocity[j] + (1 - beta2) * g * g;
      const mHat = this.biasMomentum[j] / bc1;
      const vHat = this.biasVelocity[j] / bc2;
      languageModelingHead.bias[j] *= 1 - learningRate * weightDecay;
      languageModelingHead.bias[j] -= learningRate * (mHat / (Math.sqrt(vHat) + eps));
    }

    return loss;
  }
}
