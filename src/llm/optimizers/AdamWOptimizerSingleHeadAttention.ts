import type { Optimizer } from './utils.ts';

import { BigramLanguageModelSingleHeadAttention } from '../models/BigramLanguageModelSingleHeadAttention.ts';
import { matrixMultiply, softmax, type Tensor1d, type Tensor2d, transpose } from '../tensorOps.ts';

export class AdamWOptimizerSingleHeadAttention implements Optimizer {
  private readonly beta1: number;
  private readonly beta2: number;

  private readonly biasMomentum: Tensor1d;
  private readonly biasVelocity: Tensor1d;
  private readonly embeddingMomentum: Tensor2d;
  private readonly embeddingVelocity: Tensor2d;
  private readonly eps: number;
  private readonly learningRate: number;

  private readonly model: BigramLanguageModelSingleHeadAttention;
  private stepCount = 0;
  private readonly weightDecay: number;
  private readonly weightsMomentum: Tensor2d;

  private readonly weightsVelocity: Tensor2d;

  private readonly wkMomentum: Tensor2d;

  private readonly wkVelocity: Tensor2d;

  private readonly wqMomentum: Tensor2d;

  private readonly wqVelocity: Tensor2d;

  private readonly wvMomentum: Tensor2d;

  private readonly wvVelocity: Tensor2d;

  constructor(
    model: BigramLanguageModelSingleHeadAttention,
    learningRate: number,
    beta1: number,
    beta2: number,
    eps: number,
    weightDecay: number,
  ) {
    this.weightDecay = weightDecay;
    this.eps = eps;
    this.beta2 = beta2;
    this.beta1 = beta1;
    this.learningRate = learningRate;
    this.model = model;
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

  async train(contextTokens: Tensor2d, targets: Tensor2d): Promise<number> {
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
