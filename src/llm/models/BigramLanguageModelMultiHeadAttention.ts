import type { LanguageModel, Parameter } from '@/llm/types.ts';

import { random } from '@/lib/random.ts';
import {
  matrixMultiply,
  softmax,
  sum2d,
  type Tensor1d,
  type Tensor2d,
  type Tensor3d,
  transpose,
} from '@/llm/tensorOps.ts';

import { Linear } from './Linear.ts';
import { MultiHeadAttention } from './MultiHeadAttention.ts';
import { concatBatched, crossEntropy, sampleMultinomial, softmaxBatched } from './utils.ts';

export interface BigramLanguageModelMultiHeadAttentionSerializedData {
  vocabSize: number;
  numberEmbeddingDimensions: number;
  contextSize: number;
  numHeads: number;
  tokenEmbeddingTable: Tensor2d;
  positionEmbeddingTable: Tensor2d;
  multiHeadAttention: {
    key: { weights: Tensor2d };
    query: { weights: Tensor2d };
    value: { weights: Tensor2d };
  }[];
  languageModelingHead: { bias: Tensor1d; weights: Tensor2d };
}

export class BigramLanguageModelMultiHeadAttention implements LanguageModel<BigramLanguageModelMultiHeadAttentionSerializedData> {
  readonly contextSize: number;

  readonly languageModelingHead: Linear;
  readonly multiHeadAttention: MultiHeadAttention;
  readonly positionEmbeddingTable: Tensor2d;
  readonly tokenEmbeddingTable: Tensor2d;
  constructor(vocabSize: number, numberEmbeddingDimensions: number, contextSize: number, numHeads: number) {
    this.contextSize = contextSize;
    this.tokenEmbeddingTable = Array.from({ length: vocabSize }, () =>
      Array.from({ length: numberEmbeddingDimensions }, () => random() * 0.01),
    );
    this.positionEmbeddingTable = Array.from({ length: contextSize }, () =>
      Array.from({ length: numberEmbeddingDimensions }, () => random() * 0.01),
    );

    this.multiHeadAttention = new MultiHeadAttention(numberEmbeddingDimensions, numHeads);
    this.languageModelingHead = new Linear(numberEmbeddingDimensions, vocabSize);
  }

  static fromSerializedData(
    data: BigramLanguageModelMultiHeadAttentionSerializedData,
  ): BigramLanguageModelMultiHeadAttention {
    const model = new BigramLanguageModelMultiHeadAttention(
      data.vocabSize,
      data.numberEmbeddingDimensions,
      data.contextSize,
      data.numHeads,
    );

    // Restore embeddings
    model.tokenEmbeddingTable.splice(0, model.tokenEmbeddingTable.length, ...data.tokenEmbeddingTable);
    model.positionEmbeddingTable.splice(0, model.positionEmbeddingTable.length, ...data.positionEmbeddingTable);

    // Restore attention heads
    data.multiHeadAttention.forEach((headData, i) => {
      const head = model.multiHeadAttention.heads[i];
      head.key.weights.splice(0, head.key.weights.length, ...headData.key.weights);
      head.query.weights.splice(0, head.query.weights.length, ...headData.query.weights);
      head.value.weights.splice(0, head.value.weights.length, ...headData.value.weights);
    });

    // Restore language modeling head
    model.languageModelingHead.weights.splice(
      0,
      model.languageModelingHead.weights.length,
      ...data.languageModelingHead.weights,
    );
    model.languageModelingHead.bias.splice(
      0,
      model.languageModelingHead.bias.length,
      ...data.languageModelingHead.bias,
    );

    return model;
  }

  computeGradients(contextTokens: Tensor2d, targets: Tensor2d): Record<string, Tensor2d | Tensor1d> {
    const B = contextTokens.length;
    const T = contextTokens[0].length;
    const scale = 1 / (B * T);

    // Forward pass (save intermediates for backward)
    const tokenEmbeddings = contextTokens.map((batch) => batch.map((token) => this.tokenEmbeddingTable[token])); // (B, T, C)
    const positionEmbeddings = contextTokens[0].map((_, i) => this.positionEmbeddingTable[i]); // (T, C)
    const embeddingsSum = tokenEmbeddings.map((batch) => sum2d(batch, positionEmbeddings)); // (B, T, C)
    const attended = this.multiHeadAttention.forward(embeddingsSum); // (B, T, C)
    const logits = attended.map((batch) => batch.map((token) => this.languageModelingHead.forward(token))); // (B, T, vocabSize)

    // Initialize gradient accumulators
    const gradients: Record<string, Tensor2d | Tensor1d> = {
      tokenEmbedding: this.tokenEmbeddingTable.map((row) => new Array<number>(row.length).fill(0)),
      positionEmbedding: this.positionEmbeddingTable.map((row) => new Array<number>(row.length).fill(0)),
      lmWeights: this.languageModelingHead.weights.map((row) => new Array<number>(row.length).fill(0)),
      lmBias: new Array<number>(this.languageModelingHead.bias.length).fill(0),
    };

    // Initialize head gradients
    this.multiHeadAttention.heads.forEach((head, headIdx) => {
      gradients[`head${headIdx}_key`] = head.key.weights.map((row) => new Array<number>(row.length).fill(0));
      gradients[`head${headIdx}_query`] = head.query.weights.map((row) => new Array<number>(row.length).fill(0));
      gradients[`head${headIdx}_value`] = head.value.weights.map((row) => new Array<number>(row.length).fill(0));
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
        for (let i = 0; i < attended[b][t].length; i++) {
          for (let j = 0; j < dLogits[t].length; j++) {
            (gradients.lmWeights as Tensor2d)[i][j] += attended[b][t][i] * dLogits[t][j];
          }
        }
        for (let j = 0; j < dLogits[t].length; j++) {
          (gradients.lmBias as Tensor1d)[j] += dLogits[t][j];
        }
      }

      // d_attended = dLogits @ Wlm^T
      const dAttended = matrixMultiply(dLogits, transpose(this.languageModelingHead.weights));

      // Backward through multi-head attention
      const { dX, headGrads } = this.multiHeadAttention.backward(embeddingsSum[b], dAttended);

      // Accumulate head gradients
      headGrads.forEach((headGrad, headIdx) => {
        const { dWk, dWq, dWv } = headGrad;
        for (let i = 0; i < dWk.length; i++) {
          for (let j = 0; j < dWk[i].length; j++) {
            (gradients[`head${headIdx}_key`] as Tensor2d)[i][j] += dWk[i][j];
            (gradients[`head${headIdx}_query`] as Tensor2d)[i][j] += dWq[i][j];
            (gradients[`head${headIdx}_value`] as Tensor2d)[i][j] += dWv[i][j];
          }
        }
      });

      // Backward through embedding lookup
      for (let t = 0; t < T; t++) {
        const token = contextTokens[b][t];
        for (let i = 0; i < dX[t].length; i++) {
          (gradients.tokenEmbedding as Tensor2d)[token][i] += dX[t][i];
          (gradients.positionEmbedding as Tensor2d)[t][i] += dX[t][i];
        }
      }
    }

    return gradients;
  }

  forward(
    idx: Tensor2d, // (B, T)
    targets?: Tensor2d, // (B, T)
  ): Promise<{
    logits: Tensor3d; // (B, T, C)
    loss?: number;
  }> {
    const tokenEmbeddings = idx.map((batch) => batch.map((token) => this.tokenEmbeddingTable[token])); // (B, T, C)
    const positionEmbeddings = idx[0].map((_, i) => this.positionEmbeddingTable[i]); // (T, C)
    const embeddingsSum = tokenEmbeddings.map((batch) => sum2d(batch, positionEmbeddings)); // (B, T, C)
    const attended = this.multiHeadAttention.forward(embeddingsSum); // (B, T, C)
    const logits = attended.map((batch) => batch.map((token) => this.languageModelingHead.forward(token))); // (B, T, vocabSize)

    if (!targets) return Promise.resolve({ logits });
    const loss = crossEntropy(logits, targets);

    return Promise.resolve({ logits, loss });
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
      { name: 'lmWeights', data: this.languageModelingHead.weights },
      { name: 'lmBias', data: this.languageModelingHead.bias },
    ];

    // Add parameters from each attention head
    this.multiHeadAttention.heads.forEach((head, headIdx) => {
      params.push(
        { name: `head${headIdx}_key`, data: head.key.weights },
        { name: `head${headIdx}_query`, data: head.query.weights },
        { name: `head${headIdx}_value`, data: head.value.weights },
      );
    });

    return params;
  }

  getSerializedData(): BigramLanguageModelMultiHeadAttentionSerializedData {
    return {
      vocabSize: this.tokenEmbeddingTable.length,
      numberEmbeddingDimensions: this.tokenEmbeddingTable[0].length,
      contextSize: this.contextSize,
      numHeads: this.multiHeadAttention.heads.length,
      tokenEmbeddingTable: this.tokenEmbeddingTable,
      positionEmbeddingTable: this.positionEmbeddingTable,
      multiHeadAttention: this.multiHeadAttention.heads.map((head) => ({
        key: { weights: head.key.weights },
        query: { weights: head.query.weights },
        value: { weights: head.value.weights },
      })),
      languageModelingHead: {
        weights: this.languageModelingHead.weights,
        bias: this.languageModelingHead.bias,
      },
    };
  }
}
