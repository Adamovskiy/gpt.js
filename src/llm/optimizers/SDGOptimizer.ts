import { softmax, type Tensor2d } from '../tensorOps.ts';
import type { Model, Optimizer } from './utils.ts';

export class SDGOptimizer implements Optimizer {
  private readonly learningRate: number;

  private readonly model: Model;

  constructor(model: Model, learningRate: number) {
    this.model = model;
    this.learningRate = learningRate;
  }

  async train(contextTokens: Tensor2d, outputs: Tensor2d): Promise<number> {
    const { logits, loss } = await this.model.forward(contextTokens, outputs);

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
