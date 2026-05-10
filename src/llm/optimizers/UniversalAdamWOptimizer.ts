import type { Tensor1d, Tensor2d } from '../tensorOps.ts';
import type { Trainable } from '../types.ts';
import type { Optimizer } from './utils.ts';

export class UniversalAdamWOptimizer implements Optimizer {
  private readonly beta1: number = 0.9;
  private readonly beta2: number = 0.999;
  private readonly eps: number = 1e-8;

  private readonly learningRate: number;

  private readonly model: Trainable;

  private momentum: Record<string, Tensor2d | Tensor1d> = {};

  private stepCount = 0;

  private velocity: Record<string, Tensor2d | Tensor1d> = {};

  private readonly weightDecay: number = 0.01;

  constructor(model: Trainable, learningRate: number, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0.01) {
    this.weightDecay = weightDecay;
    this.eps = eps;
    this.beta2 = beta2;
    this.beta1 = beta1;
    this.learningRate = learningRate;
    this.model = model;
    // Initialize momentum and velocity for all parameters
    const params = this.model.getParameters();
    for (const param of params) {
      if (Array.isArray(param.data[0])) {
        // 2D parameter
        const data = param.data as Tensor2d;
        this.momentum[param.name] = data.map((row) => new Array<number>(row.length).fill(0));
        this.velocity[param.name] = data.map((row) => new Array<number>(row.length).fill(0));
      } else {
        // 1D parameter
        const data = param.data as Tensor1d;
        this.momentum[param.name] = new Array<number>(data.length).fill(0);
        this.velocity[param.name] = new Array<number>(data.length).fill(0);
      }
    }
  }

  async train(contextTokens: Tensor2d, targets: Tensor2d): Promise<number> {
    this.stepCount++;

    // Forward pass first to get loss and logits
    const { logits, loss } = await this.model.forward(contextTokens, targets);

    // Compute gradients (may reuse forward pass results for GPU models)
    const gradients = this.model.computeGradients(contextTokens, targets, logits);

    // AdamW update
    const { beta1, beta2, eps, learningRate, weightDecay } = this;
    const bc1 = 1 - Math.pow(beta1, this.stepCount);
    const bc2 = 1 - Math.pow(beta2, this.stepCount);

    const params = this.model.getParameters();
    for (const param of params) {
      const grad = gradients[param.name];
      const momentum = this.momentum[param.name];
      const velocity = this.velocity[param.name];

      if (Array.isArray(param.data[0])) {
        // 2D parameter
        const data = param.data as Tensor2d;
        const gradData = grad as Tensor2d;
        const momentumData = momentum as Tensor2d;
        const velocityData = velocity as Tensor2d;

        for (let i = 0; i < data.length; i++) {
          for (let j = 0; j < data[i].length; j++) {
            const g = gradData[i][j];
            momentumData[i][j] = beta1 * momentumData[i][j] + (1 - beta1) * g;
            velocityData[i][j] = beta2 * velocityData[i][j] + (1 - beta2) * g * g;
            const mHat = momentumData[i][j] / bc1;
            const vHat = velocityData[i][j] / bc2;
            data[i][j] *= 1 - learningRate * weightDecay;
            data[i][j] -= learningRate * (mHat / (Math.sqrt(vHat) + eps));
          }
        }
      } else {
        // 1D parameter
        const data = param.data as Tensor1d;
        const gradData = grad as Tensor1d;
        const momentumData = momentum as Tensor1d;
        const velocityData = velocity as Tensor1d;

        for (let i = 0; i < data.length; i++) {
          const g = gradData[i];
          momentumData[i] = beta1 * momentumData[i] + (1 - beta1) * g;
          velocityData[i] = beta2 * velocityData[i] + (1 - beta2) * g * g;
          const mHat = momentumData[i] / bc1;
          const vHat = velocityData[i] / bc2;
          data[i] *= 1 - learningRate * weightDecay;
          data[i] -= learningRate * (mHat / (Math.sqrt(vHat) + eps));
        }
      }
    }

    return loss || 0;
  }
}
