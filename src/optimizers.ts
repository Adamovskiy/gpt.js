import { softmax, type Tensor1d, type Tensor2d, type Tensor3d } from './tensorOps.js';
import type { Linear } from './tfOps.js';

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
