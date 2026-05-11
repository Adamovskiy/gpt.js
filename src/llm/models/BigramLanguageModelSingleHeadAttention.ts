import type { LanguageModel, Parameter } from '@/llm/types.ts';

import { random } from '@/lib/random.ts';
import {
  matrixMultiply,
  softmax,
  sum1d,
  sum2d,
  type Tensor1d,
  type Tensor2d,
  type Tensor3d,
  transpose,
} from '@/llm/tensorOps.ts';

import { Head } from './Head.ts';
import { Linear } from './Linear.ts';
import { concatBatched, crossEntropy, sampleMultinomial, softmaxBatched } from './utils.ts';

export class BigramLanguageModelSingleHeadAttention implements LanguageModel {
  readonly contextSize: number;
  readonly languageModelingHead: Linear; // Transforms attended embeddings to logits
  readonly positionEmbeddingTable: Tensor2d; // blockSize x numberEmbeddingDimensions
  readonly selfAttention: Head; // single self-attention head
  readonly tokenEmbeddingTable: Tensor2d; // vocabSize x numberEmbeddingDimensions

  constructor(vocabSize: number, numberEmbeddingDimensions: number, contextSize: number) {
    this.contextSize = contextSize;
    this.tokenEmbeddingTable = Array.from({ length: vocabSize }, () =>
      Array.from({ length: numberEmbeddingDimensions }, () => random() * 0.01),
    );
    this.positionEmbeddingTable = Array.from({ length: contextSize }, () =>
      Array.from({ length: numberEmbeddingDimensions }, () => random() * 0.01),
    );

    this.selfAttention = new Head(numberEmbeddingDimensions, numberEmbeddingDimensions);
    this.languageModelingHead = new Linear(numberEmbeddingDimensions, vocabSize);
  }

  computeGradients(contextTokens: Tensor2d, targets: Tensor2d): Record<string, Tensor2d | Tensor1d> {
    // Simple gradient computation - similar to bigram but with attention
    const B = contextTokens.length;
    const T = contextTokens[0].length;
    const scale = 1 / (B * T);

    const gradients: Record<string, Tensor2d | Tensor1d> = {
      tokenEmbedding: this.tokenEmbeddingTable.map((row) => new Array<number>(row.length).fill(0)),
      positionEmbedding: this.positionEmbeddingTable.map((row) => new Array<number>(row.length).fill(0)),
      selfAttn_key: this.selfAttention.key.weights.map((row) => new Array<number>(row.length).fill(0)),
      selfAttn_query: this.selfAttention.query.weights.map((row) => new Array<number>(row.length).fill(0)),
      selfAttn_value: this.selfAttention.value.weights.map((row) => new Array<number>(row.length).fill(0)),
      lmWeights: this.languageModelingHead.weights.map((row) => new Array<number>(row.length).fill(0)),
      lmBias: new Array<number>(this.languageModelingHead.bias.length).fill(0),
    };

    for (let b = 0; b < B; b++) {
      // Forward pass to get intermediates
      const tokenEmbeddings = contextTokens[b].map((token) => this.tokenEmbeddingTable[token]);
      const positionEmbeddings = contextTokens[b].map((_, i) => this.positionEmbeddingTable[i]);
      const embeddingsSum = tokenEmbeddings.map((tokenEmb, i) => sum1d(tokenEmb, positionEmbeddings[i]));
      const attended = this.selfAttention.forward([embeddingsSum])[0];
      const logits = attended.map((emb) => this.languageModelingHead.forward(emb));

      const dLogits = logits.map((tokenLogits, t) => {
        const probs = softmax(tokenLogits);
        probs[targets[b][t]] -= 1;
        return probs.map((v) => v * scale);
      });

      // Backward through language modeling head
      for (let t = 0; t < T; t++) {
        for (let i = 0; i < attended[t].length; i++) {
          for (let j = 0; j < dLogits[t].length; j++) {
            (gradients.lmWeights as Tensor2d)[i][j] += attended[t][i] * dLogits[t][j];
          }
        }
        for (let j = 0; j < dLogits[t].length; j++) {
          (gradients.lmBias as Tensor1d)[j] += dLogits[t][j];
        }
      }

      // Simple approximation for attention gradients (could be more accurate)
      const dAttended = dLogits.map(
        (dLogit) => matrixMultiply([dLogit], transpose(this.languageModelingHead.weights))[0],
      );

      // Simplified gradient computation for embeddings
      for (let t = 0; t < T; t++) {
        const token = contextTokens[b][t];
        for (let i = 0; i < dAttended[t].length; i++) {
          (gradients.tokenEmbedding as Tensor2d)[token][i] += dAttended[t][i] * 0.5; // simplified
          (gradients.positionEmbedding as Tensor2d)[t][i] += dAttended[t][i] * 0.5;
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
    const tokenEmbeddings = idx.map((batch) => batch.map((token) => this.tokenEmbeddingTable[token])); // (B, T, numberEmbeddingDimensions)
    const positionEmbeddings = idx[0].map((_, i) => this.positionEmbeddingTable[i]); // (T, numberEmbeddingDimensions)
    const embeddingsSum = tokenEmbeddings.map((batch) => sum2d(batch, positionEmbeddings)); // (B, T, numberEmbeddingDimensions)
    const attended = this.selfAttention.forward(embeddingsSum); // (B, T, numberEmbeddingDimensions)
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
      const idxCond = idx.map((batch) => batch.slice(-this.contextSize)); // crop to blockSize
      const { logits } = await this.forward(idxCond);

      const lastTokenLogits = logits.map((batch) => batch[batch.length - 1]); // (B, C)
      const probs = softmaxBatched(lastTokenLogits); // (B, C)
      const idxNext = sampleMultinomial(probs);
      concatBatched(idx, idxNext);
    }

    return idx;
  }

  getParameters(): Parameter[] {
    return [
      { name: 'tokenEmbedding', data: this.tokenEmbeddingTable },
      { name: 'positionEmbedding', data: this.positionEmbeddingTable },
      { name: 'selfAttn_key', data: this.selfAttention.key.weights },
      { name: 'selfAttn_query', data: this.selfAttention.query.weights },
      { name: 'selfAttn_value', data: this.selfAttention.value.weights },
      { name: 'lmWeights', data: this.languageModelingHead.weights },
      { name: 'lmBias', data: this.languageModelingHead.bias },
    ];
  }
}
