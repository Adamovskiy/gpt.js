import type { Model, Optimizer } from './utils.ts';

import { softmax, type Tensor1d, type Tensor2d } from '../tensorOps.ts';

export class AdamWOptimizer implements Optimizer {
  private readonly beta1: number;
  private readonly beta2: number;

  private readonly biasMomentum: Tensor1d;
  private readonly biasVelocity: Tensor1d;

  private readonly embeddingMomentum: Tensor2d;
  private readonly embeddingVelocity: Tensor2d;

  private readonly eps: number;

  private readonly learningRate: number;

  private readonly model: Model;

  private stepCount = 0;

  private readonly weightDecay: number;

  private readonly weightsMomentum: Tensor2d;

  private readonly weightsVelocity: Tensor2d;

  constructor(model: Model, learningRate: number, beta1: number, beta2: number, eps: number, weightDecay: number) {
    this.weightDecay = weightDecay;
    this.eps = eps;
    this.beta2 = beta2;
    this.beta1 = beta1;
    this.model = model;
    this.learningRate = learningRate;
    this.embeddingMomentum = model.tokenEmbeddingTable.map((row) => new Array<number>(row.length).fill(0));
    this.embeddingVelocity = model.tokenEmbeddingTable.map((row) => new Array<number>(row.length).fill(0));
    this.weightsMomentum = model.languageModelingHead.weights.map((row) => new Array<number>(row.length).fill(0));
    this.weightsVelocity = model.languageModelingHead.weights.map((row) => new Array<number>(row.length).fill(0));
    this.biasMomentum = new Array<number>(model.languageModelingHead.bias.length).fill(0);
    this.biasVelocity = new Array<number>(model.languageModelingHead.bias.length).fill(0);
  }

  async train(contextTokens: Tensor2d, outputs: Tensor2d): Promise<number> {
    this.stepCount++;
    const { logits, loss } = await this.model.forward(contextTokens, outputs);

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
