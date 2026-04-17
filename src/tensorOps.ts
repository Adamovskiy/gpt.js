export type Tensor1d = number[];
export type Tensor2d = number[][];
export type Tensor3d = number[][][];

// Warning: it mutates the target!
function add(target: Tensor1d, addition: Tensor1d) {
  if (target.length !== addition.length) throw new Error('Tensors should be of the same size');

  target.forEach((el, i) => {
    target[i] = el + addition[i];
  });

  return target;
}

function divide(target: Tensor1d, divider: Tensor1d) {
  if (target.length !== divider.length) throw new Error('Tensors should be of the same size');

  target.forEach((el, i) => {
    target[i] = el / divider[i];
  });

  return target;
}

export function lowerTriangularMatrixAvgWeighted(size: number): Tensor2d {
  return Array.from({ length: size }, (_, i) => Array.from({ length: size }, (_, j) => (j < i + 1 ? 1 / (i + 1) : 0)));
}

export function softmax(logits: Tensor1d) {
  const maxLogit = Math.max(...logits);
  const exp = logits.map((logit) => Math.exp(logit - maxLogit));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((exp) => exp / sum);
}

export function lowerTriangularMatrixAvgWeightedSoftmax(size: number): Tensor2d {
  // How much each token is interested in another
  // -infinity means no communication
  const affinitiesMatrix = Array.from({ length: size }, (_, i) =>
    // The same affinity to all previous tokens, no affinity to future ones
    Array.from({ length: size }, (_, j) => (j < i + 1 ? 0 : -Infinity)),
  );

  // Transform zeros\-infinities triangle to weighted 1\0 triangle
  return affinitiesMatrix.map(softmax);
}

export function matrixMultiply(a: Tensor2d, b: Tensor2d) {
  if (a[0].length !== b.length) throw new Error('Matrices should be of the compatible sizes');

  return a.map((row) => {
    return b[0].map((_, i) => {
      return row.reduce((acc, curr, j) => acc + curr * b[j][i], 0);
    });
  });
}

export function getBagOfWordsUnoptimized(logits: Tensor3d) {
  // Averages of previous tokens (including the current)
  return logits.map((batch) =>
    batch.map((_, i) => {
      // avg of logits prior to the current one (including it)
      const prevLogitsSum = batch
        .slice(0, i + 1)
        .reduce((acc, curr) => add(acc, curr), new Array<number>(batch[0].length).fill(0));

      return divide(prevLogitsSum, new Array<number>(prevLogitsSum.length).fill(i + 1));
    }),
  );
}

export function getBagOfWordsOptimized(logits: Tensor3d) {
  return logits.map((batch) => matrixMultiply(lowerTriangularMatrixAvgWeightedSoftmax(batch.length), batch));
}
