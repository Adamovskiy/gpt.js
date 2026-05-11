import { it } from 'vitest';

import { assertArraysClose, assertMatrix2dClose, assertMatrix3dClose } from '@/testUtils.ts';

import {
  getBagOfWordsOptimized,
  getBagOfWordsUnoptimized,
  lowerTriangularMatrixAvgWeighted,
  lowerTriangularMatrixAvgWeightedSoftmax,
  matrixMultiply,
  softmax,
  sum2d,
} from './tensorOps.ts';

it('sum2d', () => {
  // Sum of tensors is tensor of sums of its components
  const a = [
    [1, 2],
    [3, 4],
  ];
  const b = [
    [0.5, 0.6],
    [0.7, 0.8],
  ];
  const sum = [
    [1.5, 2.6],
    [3.7, 4.8],
  ];
  assertMatrix2dClose(sum2d(a, b), sum);
});

it('matrixMultiply', () => {
  const a = [
    [0, 0.5, 1],
    [2, 3, 4],
    [5, 6, 7],
  ];
  const b = [
    [0, 1],
    [2, 3],
    [4, 5],
  ];
  const ab = [
    [5, 6.5],
    [22, 31],
    [40, 58],
  ];
  assertMatrix2dClose(matrixMultiply(a, b), ab);
});

it('softmax', () => {
  // Transforms -Infinity logit to 0 probability
  assertArraysClose(softmax([0, -Infinity]), [1, 0]);
  // Transforms same logits as equal probabilities
  assertArraysClose(softmax([42, 42, 42]), [1 / 3, 1 / 3, 1 / 3]);
  // Transforms logits to normalized exponential probabilities
  assertArraysClose(softmax([0, 1, 100]), [3e-44, 3e-43, 1]);
});

it('lowerTriangularMatrixAvgWeighted', () => {
  // Top right triangle is zeros, left bottom is average distributions
  assertMatrix2dClose(lowerTriangularMatrixAvgWeighted(3), [
    [1, 0, 0],
    [0.5, 0.5, 0],
    [1 / 3, 1 / 3, 1 / 3],
  ]);
});

it('lowerTriangularMatrixAvgWeightedSoftmax', () => {
  // Does the same as lowerTriangularMatrixAvgWeighted, but using softmax inside
  assertMatrix2dClose(lowerTriangularMatrixAvgWeightedSoftmax(3), lowerTriangularMatrixAvgWeighted(3));
});

it('getBagOfWordsUnoptimized', () => {
  // Batches of logits of tokens
  const input = [
    [
      [2, 7], // "Expectation measure" of token 1 at position 0 is 2, of token 2 - 7
      [6, 4],
      [6, 5],
    ],
  ];
  // For each position, averages previous tokens' logits
  const expectedOutput = [
    [
      [2, 7],
      [4, 5.5],
      [14 / 3, 16 / 3],
    ],
  ];
  assertMatrix3dClose(getBagOfWordsUnoptimized(input), expectedOutput);
});

it('getBagOfWordsOptimized', () => {
  // Each batch is processed independently
  const input = [
    [
      [1, 0, 0],
      [0, 2, 0],
      [0, 0, 3],
      [1, 1, 1],
    ],
    [
      [3, 1, 4],
      [1, 5, 9],
      [2, 6, 5],
      [3, 5, 8],
    ],
  ];
  // The result is the same, but using softmax inside
  assertMatrix3dClose(getBagOfWordsOptimized(input), getBagOfWordsUnoptimized(input));
});
