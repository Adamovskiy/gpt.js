import type { Tensor1d, Tensor2d } from '@/llm/tensorOps.ts';
import type { Optimizer, Trainable } from '@/llm/types.ts';

export interface SDGOptimizerSerializedData {
  learningRate: number;
}

export class SDGOptimizer implements Optimizer {
  private readonly learningRate: number;

  private readonly model: Trainable;

  constructor(model: Trainable, learningRate: number) {
    this.model = model;
    this.learningRate = learningRate;
  }

  static fromSerializedData(data: SDGOptimizerSerializedData, model: Trainable): SDGOptimizer {
    return new SDGOptimizer(model, data.learningRate);
  }

  getSerializedData(): SDGOptimizerSerializedData {
    return {
      learningRate: this.learningRate,
    };
  }

  async train(contextTokens: Tensor2d, targets: Tensor2d): Promise<number> {
    // Forward pass to get loss and logits
    const { logits, loss } = await this.model.forward(contextTokens, targets);

    // Compute gradients using the model's computeGradients method
    const gradients = this.model.computeGradients(contextTokens, targets, logits);

    // Apply SGD updates to all parameters
    const params = this.model.getParameters();
    for (const param of params) {
      const grad = gradients[param.name];

      if (Array.isArray(param.data[0])) {
        // 2D parameter
        const data = param.data as Tensor2d;
        const gradData = grad as Tensor2d;

        for (let i = 0; i < data.length; i++) {
          for (let j = 0; j < data[i].length; j++) {
            data[i][j] -= this.learningRate * gradData[i][j];
          }
        }
      } else {
        // 1D parameter
        const data = param.data as Tensor1d;
        const gradData = grad as Tensor1d;

        for (let i = 0; i < data.length; i++) {
          data[i] -= this.learningRate * gradData[i];
        }
      }
    }

    return loss ?? 0;
  }
}
