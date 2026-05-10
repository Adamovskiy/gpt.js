import type { LanguageModel, Parameter } from '../types.ts';
import {
  matrixMultiply,
  softmax,
  sum1d,
  sum2d,
  type Tensor1d,
  type Tensor2d,
  type Tensor3d,
  transpose,
} from '../tensorOps.ts';
import { Linear } from './Linear.ts';
import { random } from '../../lib/random.ts';
import { concatBatched, crossEntropy, sampleMultinomial, softmaxBatched } from './utils.ts';

export class BigramLanguageModel implements LanguageModel {
  readonly tokenEmbeddingTable: Tensor2d; // vocabSize x numberEmbeddingDimensions
  readonly positionEmbeddingTable: Tensor2d; // blockSize x numberEmbeddingDimensions
  readonly languageModelingHead: Linear; // Transforms embeddings to logits
  readonly contextSize: number;

  constructor(vocabSize: number, numberEmbeddingDimensions: number, contextSize: number) {
    this.contextSize = contextSize;
    this.tokenEmbeddingTable = Array.from({ length: vocabSize }, () =>
      Array.from({ length: numberEmbeddingDimensions }, () => random() * 0.01),
    );
    this.positionEmbeddingTable = Array.from({ length: contextSize }, () =>
      Array.from({ length: numberEmbeddingDimensions }, () => random() * 0.01),
    );

    this.languageModelingHead = new Linear(numberEmbeddingDimensions, vocabSize);
  }

  async forward(
    idx: Tensor2d, // (B, T)
    targets?: Tensor2d, // (B, T)
  ): Promise<{
    logits: Tensor3d; // (B, T, C)
    loss?: number;
  }> {
    const tokenEmbeddings = idx.map((batch) => batch.map((token) => this.tokenEmbeddingTable[token])); // (B,T, numberEmbeddingDimensions)
    const positionEmbeddings = idx[0].map((_, i) => this.positionEmbeddingTable[i]); // (T, numberEmbeddingDimensions)
    const embeddingsSum = tokenEmbeddings.map((batch) => sum2d(batch, positionEmbeddings)); // (T, numberEmbeddingDimensions)
    const logits = embeddingsSum.map((batch) => batch.map((token) => this.languageModelingHead.forward(token))); // (B,T, vocabSize)

    if (!targets) return { logits };
    const loss = crossEntropy(logits, targets);

    return { logits, loss };
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
      { name: 'lmWeights', data: this.languageModelingHead.weights },
      { name: 'lmBias', data: this.languageModelingHead.bias },
    ];
  }

  computeGradients(
    contextTokens: Tensor2d,
    targets: Tensor2d,
  ): {
    [paramName: string]: Tensor2d | Tensor1d;
  } {
    // Simple gradient computation for bigram model
    const B = contextTokens.length;
    const T = contextTokens[0].length;
    const scale = 1 / (B * T);

    const gradients: { [paramName: string]: Tensor2d | Tensor1d } = {
      tokenEmbedding: this.tokenEmbeddingTable.map((row) => new Array<number>(row.length).fill(0)),
      positionEmbedding: this.positionEmbeddingTable.map((row) => new Array<number>(row.length).fill(0)),
      lmWeights: this.languageModelingHead.weights.map((row) => new Array<number>(row.length).fill(0)),
      lmBias: new Array<number>(this.languageModelingHead.bias.length).fill(0),
    };

    for (let b = 0; b < B; b++) {
      const tokenEmbeddings = contextTokens[b].map((token) => this.tokenEmbeddingTable[token]);
      const positionEmbeddings = contextTokens[b].map((_, i) => this.positionEmbeddingTable[i]);
      const embeddingsSum = tokenEmbeddings.map((tokenEmb, i) => sum1d(tokenEmb, positionEmbeddings[i]));
      const logits = embeddingsSum.map((emb) => this.languageModelingHead.forward(emb));

      const dLogits = logits.map((tokenLogits, t) => {
        const probs = softmax(tokenLogits);
        probs[targets[b][t]] -= 1;
        return probs.map((v) => v * scale);
      });

      // Backward through language modeling head
      for (let t = 0; t < T; t++) {
        for (let i = 0; i < embeddingsSum[t].length; i++) {
          for (let j = 0; j < dLogits[t].length; j++) {
            (gradients['lmWeights'] as Tensor2d)[i][j] += embeddingsSum[t][i] * dLogits[t][j];
          }
        }
        for (let j = 0; j < dLogits[t].length; j++) {
          (gradients['lmBias'] as Tensor1d)[j] += dLogits[t][j];
        }
      }

      // Backward through embeddings
      const dEmbeddings = dLogits.map(
        (dLogit) => matrixMultiply([dLogit], transpose(this.languageModelingHead.weights))[0],
      );

      for (let t = 0; t < T; t++) {
        const token = contextTokens[b][t];
        for (let i = 0; i < dEmbeddings[t].length; i++) {
          (gradients['tokenEmbedding'] as Tensor2d)[token][i] += dEmbeddings[t][i];
          (gradients['positionEmbedding'] as Tensor2d)[t][i] += dEmbeddings[t][i];
        }
      }
    }

    return gradients;
  }
}
