import type { LanguageModel, Parameter } from '../types.ts';
import {
  matrixMultiply,
  softmax,
  sum2d,
  type Tensor1d,
  type Tensor2d,
  type Tensor3d,
  transpose,
} from '../tensorOps.ts';
import { TransformerBlock } from './TransformerBlock.ts';
import { LayerNorm } from './LayerNorm.ts';
import { Linear } from './Linear.ts';
import { random } from '../../lib/random.ts';
import { concatBatched, crossEntropy, sampleMultinomial, softmaxBatched } from './utils.ts';

export class GPTModel implements LanguageModel {
  readonly tokenEmbeddingTable: Tensor2d;
  readonly positionEmbeddingTable: Tensor2d;
  readonly blocks: TransformerBlock[];
  readonly lnFinal: LayerNorm;
  readonly languageModelingHead: Linear;
  readonly contextSize: number;

  get isGPU(): boolean {
    return false;
  }

  constructor(
    vocabSize: number,
    numberEmbeddingDimensions: number,
    contextSize: number,
    numHeads: number,
    numLayers: number,
  ) {
    this.contextSize = contextSize;
    this.tokenEmbeddingTable = Array.from({ length: vocabSize }, () =>
      Array.from({ length: numberEmbeddingDimensions }, () => random() * 0.01),
    );
    this.positionEmbeddingTable = Array.from({ length: contextSize }, () =>
      Array.from({ length: numberEmbeddingDimensions }, () => random() * 0.01),
    );

    // Create multiple transformer blocks
    this.blocks = Array.from({ length: numLayers }, () => new TransformerBlock(numberEmbeddingDimensions, numHeads));
    this.lnFinal = new LayerNorm(numberEmbeddingDimensions);
    this.languageModelingHead = new Linear(numberEmbeddingDimensions, vocabSize);
  }

  async forward(
    idx: Tensor2d, // (B, T)
    targets?: Tensor2d, // (B, T)
  ): Promise<{
    logits: Tensor3d; // (B, T, C)
    loss?: number;
  }> {
    const tokenEmbeddings = idx.map((batch) => batch.map((token) => this.tokenEmbeddingTable[token])); // (B, T, C)
    const positionEmbeddings = idx[0].map((_, i) => this.positionEmbeddingTable[i]); // (T, C)
    let x = tokenEmbeddings.map((batch) => sum2d(batch, positionEmbeddings)); // (B, T, C)

    // Pass through all transformer blocks
    for (const block of this.blocks) {
      x = block.forward(x);
    }

    // Final layer norm and language modeling head
    const normalized = this.lnFinal.forward(x);
    const logits = normalized.map((batch) => batch.map((token) => this.languageModelingHead.forward(token))); // (B, T, vocabSize)

    if (!targets) return { logits };
    const loss = crossEntropy(logits, targets);

    return { logits, loss };
  }

  async generate(
    idx: Tensor2d, // (B, T, C)
    maxNewTokens: number,
  ): Promise<Tensor2d> {
    for (let i = 0; i < maxNewTokens; i++) {
      const idxCond = idx.map((batch) => batch.slice(-this.contextSize)); // crop to contextSize
      const { logits } = await this.forward(idxCond);

      const lastTokenLogits = logits.map((batch) => batch[batch.length - 1]); // (B, C)
      const probs = softmaxBatched(lastTokenLogits); // (B, C)
      const idxNext = sampleMultinomial(probs);
      concatBatched(idx, idxNext);
    }

    return idx;
  }

  getParameters(): Parameter[] {
    const params: Parameter[] = [
      { name: 'tokenEmbedding', data: this.tokenEmbeddingTable },
      { name: 'positionEmbedding', data: this.positionEmbeddingTable },
      { name: 'lnFinal_gamma', data: this.lnFinal.gamma },
      { name: 'lnFinal_beta', data: this.lnFinal.beta },
      { name: 'lmWeights', data: this.languageModelingHead.weights },
      { name: 'lmBias', data: this.languageModelingHead.bias },
    ];

    // Add parameters from all transformer blocks
    this.blocks.forEach((block, layerIdx) => {
      // LayerNorm 1
      params.push(
        { name: `layer${layerIdx}_ln1_gamma`, data: block.ln1.gamma },
        { name: `layer${layerIdx}_ln1_beta`, data: block.ln1.beta },
      );

      // Attention heads
      block.multiHeadAttention.heads.forEach((head, headIdx) => {
        params.push(
          {
            name: `layer${layerIdx}_head${headIdx}_key`,
            data: head.key.weights,
          },
          {
            name: `layer${layerIdx}_head${headIdx}_query`,
            data: head.query.weights,
          },
          {
            name: `layer${layerIdx}_head${headIdx}_value`,
            data: head.value.weights,
          },
        );
      });

      // LayerNorm 2 & FeedForward
      params.push(
        { name: `layer${layerIdx}_ln2_gamma`, data: block.ln2.gamma },
        { name: `layer${layerIdx}_ln2_beta`, data: block.ln2.beta },
        {
          name: `layer${layerIdx}_ff1Weights`,
          data: block.feedForward.linear1.weights,
        },
        {
          name: `layer${layerIdx}_ff1Bias`,
          data: block.feedForward.linear1.bias,
        },
        {
          name: `layer${layerIdx}_ff2Weights`,
          data: block.feedForward.linear2.weights,
        },
        {
          name: `layer${layerIdx}_ff2Bias`,
          data: block.feedForward.linear2.bias,
        },
      );
    });

    return params;
  }

  computeGradients(
    contextTokens: Tensor2d,
    targets: Tensor2d,
  ): {
    [paramName: string]: Tensor2d | Tensor1d;
  } {
    const B = contextTokens.length;
    const T = contextTokens[0].length;
    const scale = 1 / (B * T);

    // Forward pass (save intermediates for backward)
    const tokenEmbeddings = contextTokens.map((batch) => batch.map((token) => this.tokenEmbeddingTable[token])); // (B, T, C)
    const positionEmbeddings = contextTokens[0].map((_, i) => this.positionEmbeddingTable[i]); // (T, C)
    let x = tokenEmbeddings.map((batch) => sum2d(batch, positionEmbeddings)); // (B, T, C)

    // Save intermediate activations for backward pass
    const activations: Tensor3d[] = [x];
    for (const block of this.blocks) {
      x = block.forward(x);
      activations.push(x);
    }

    const normalized = this.lnFinal.forward(x);
    const logits = normalized.map((batch) => batch.map((token) => this.languageModelingHead.forward(token))); // (B, T, vocabSize)

    // Initialize gradient accumulators
    const gradients: { [paramName: string]: Tensor2d | Tensor1d } = {
      tokenEmbedding: this.tokenEmbeddingTable.map((row) => new Array<number>(row.length).fill(0)),
      positionEmbedding: this.positionEmbeddingTable.map((row) => new Array<number>(row.length).fill(0)),
      lnFinal_gamma: new Array<number>(this.lnFinal.gamma.length).fill(0),
      lnFinal_beta: new Array<number>(this.lnFinal.beta.length).fill(0),
      lmWeights: this.languageModelingHead.weights.map((row) => new Array<number>(row.length).fill(0)),
      lmBias: new Array<number>(this.languageModelingHead.bias.length).fill(0),
    };

    // Initialize block gradients
    this.blocks.forEach((block, layerIdx) => {
      gradients[`layer${layerIdx}_ln1_gamma`] = new Array<number>(block.ln1.gamma.length).fill(0);
      gradients[`layer${layerIdx}_ln1_beta`] = new Array<number>(block.ln1.beta.length).fill(0);
      gradients[`layer${layerIdx}_ln2_gamma`] = new Array<number>(block.ln2.gamma.length).fill(0);
      gradients[`layer${layerIdx}_ln2_beta`] = new Array<number>(block.ln2.beta.length).fill(0);
      gradients[`layer${layerIdx}_ff1Weights`] = block.feedForward.linear1.weights.map((row) =>
        new Array<number>(row.length).fill(0),
      );
      gradients[`layer${layerIdx}_ff1Bias`] = new Array<number>(block.feedForward.linear1.bias.length).fill(0);
      gradients[`layer${layerIdx}_ff2Weights`] = block.feedForward.linear2.weights.map((row) =>
        new Array<number>(row.length).fill(0),
      );
      gradients[`layer${layerIdx}_ff2Bias`] = new Array<number>(block.feedForward.linear2.bias.length).fill(0);

      block.multiHeadAttention.heads.forEach((head, headIdx) => {
        gradients[`layer${layerIdx}_head${headIdx}_key`] = head.key.weights.map((row) =>
          new Array<number>(row.length).fill(0),
        );
        gradients[`layer${layerIdx}_head${headIdx}_query`] = head.query.weights.map((row) =>
          new Array<number>(row.length).fill(0),
        );
        gradients[`layer${layerIdx}_head${headIdx}_value`] = head.value.weights.map((row) =>
          new Array<number>(row.length).fill(0),
        );
      });
    });

    for (let b = 0; b < B; b++) {
      // d_logits = (softmax(logits) - one_hot(target)) * scale
      const dLogits = logits[b].map((tokenLogits, t) => {
        const probs = softmax(tokenLogits);
        probs[targets[b][t]] -= 1;
        return probs.map((v) => v * scale);
      }); // (T, vocabSize)

      // Backward through LM head
      for (let t = 0; t < T; t++) {
        for (let i = 0; i < normalized[b][t].length; i++) {
          for (let j = 0; j < dLogits[t].length; j++) {
            (gradients['lmWeights'] as Tensor2d)[i][j] += normalized[b][t][i] * dLogits[t][j];
          }
        }
        for (let j = 0; j < dLogits[t].length; j++) {
          (gradients['lmBias'] as Tensor1d)[j] += dLogits[t][j];
        }
      }

      // d_normalized = dLogits @ Wlm^T
      let dCurrent = matrixMultiply(dLogits, transpose(this.languageModelingHead.weights));

      // Backward through final LayerNorm
      const {
        dX: dFinalX,
        dGamma: dFinalGamma,
        dBeta: dFinalBeta,
      } = this.lnFinal.backward(activations[activations.length - 1][b], dCurrent);

      for (let i = 0; i < dFinalGamma.length; i++) {
        (gradients['lnFinal_gamma'] as Tensor1d)[i] += dFinalGamma[i];
        (gradients['lnFinal_beta'] as Tensor1d)[i] += dFinalBeta[i];
      }

      dCurrent = dFinalX;

      // Backward through transformer blocks (in reverse order)
      for (let layerIdx = this.blocks.length - 1; layerIdx >= 0; layerIdx--) {
        const block = this.blocks[layerIdx];
        const blockInput = activations[layerIdx][b];

        const { dX, ln1Grads, attnGrads, ln2Grads, ffGrads } = block.backward(blockInput, dCurrent);

        // Accumulate gradients
        for (let i = 0; i < ln1Grads.dGamma.length; i++) {
          (gradients[`layer${layerIdx}_ln1_gamma`] as Tensor1d)[i] += ln1Grads.dGamma[i];
          (gradients[`layer${layerIdx}_ln1_beta`] as Tensor1d)[i] += ln1Grads.dBeta[i];
          (gradients[`layer${layerIdx}_ln2_gamma`] as Tensor1d)[i] += ln2Grads.dGamma[i];
          (gradients[`layer${layerIdx}_ln2_beta`] as Tensor1d)[i] += ln2Grads.dBeta[i];
        }

        // FF gradients
        for (let i = 0; i < ffGrads.dW1.length; i++) {
          for (let j = 0; j < ffGrads.dW1[i].length; j++) {
            (gradients[`layer${layerIdx}_ff1Weights`] as Tensor2d)[i][j] += ffGrads.dW1[i][j];
            (gradients[`layer${layerIdx}_ff2Weights`] as Tensor2d)[i][j] += ffGrads.dW2[i][j];
          }
        }
        for (let j = 0; j < ffGrads.dB1.length; j++) {
          (gradients[`layer${layerIdx}_ff1Bias`] as Tensor1d)[j] += ffGrads.dB1[j];
        }
        for (let j = 0; j < ffGrads.dB2.length; j++) {
          (gradients[`layer${layerIdx}_ff2Bias`] as Tensor1d)[j] += ffGrads.dB2[j];
        }

        // Attention gradients
        attnGrads.forEach((headGrad, headIdx) => {
          const { dWk, dWq, dWv } = headGrad;
          for (let i = 0; i < dWk.length; i++) {
            for (let j = 0; j < dWk[i].length; j++) {
              (gradients[`layer${layerIdx}_head${headIdx}_key`] as Tensor2d)[i][j] += dWk[i][j];
              (gradients[`layer${layerIdx}_head${headIdx}_query`] as Tensor2d)[i][j] += dWq[i][j];
              (gradients[`layer${layerIdx}_head${headIdx}_value`] as Tensor2d)[i][j] += dWv[i][j];
            }
          }
        });

        dCurrent = dX;
      }

      // Backward through embedding lookup
      for (let t = 0; t < T; t++) {
        const token = contextTokens[b][t];
        for (let i = 0; i < dCurrent[t].length; i++) {
          // Token embedding gradients (sparse update)
          (gradients['tokenEmbedding'] as Tensor2d)[token][i] += dCurrent[t][i];
          // Position embedding gradients (dense update)
          (gradients['positionEmbedding'] as Tensor2d)[t][i] += dCurrent[t][i];
        }
      }
    }

    return gradients;
  }
}
