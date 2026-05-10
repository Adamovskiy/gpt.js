import { random } from '../../lib/random.ts';
import { softmax, type Tensor1d, type Tensor2d, type Tensor3d } from '../tensorOps.ts';

export function randomOutputLoss(vocabSize: number) {
  return -Math.log(1 / vocabSize);
}

export function crossEntropy(logits: Tensor3d, targets: Tensor2d) {
  let sum = 0;
  let count = 0;

  for (let batchIdx = 0; batchIdx < logits.length; batchIdx++) {
    for (let tokenIdx = 0; tokenIdx < logits[batchIdx].length; tokenIdx++) {
      // Calculate loss for each token
      const tokenLogits = logits[batchIdx][tokenIdx];
      const targetIndex = targets[batchIdx][tokenIdx];
      // TODO do not calculate all probabilities, just one
      const probabilities = softmax(tokenLogits);
      sum += -Math.log(probabilities[targetIndex] + 1e-9); // Make sure it's never log(0)
      count++;
    }
  }

  return sum / count;
}

export function sampleMultinomial(
  batches: Tensor2d, // (B, C)
) {
  return batches.map((probabilities) => {
    let sum = 0;
    const rnd = random();
    for (let idx = 0; idx < probabilities.length; idx++) {
      sum += probabilities[idx];
      if (sum > rnd) {
        return idx;
      }
    }
    return probabilities.length - 1; // Fallback
  });
}

export function concatBatched(idx: Tensor2d, idxNext: Tensor1d) {
  for (let i = 0; i < idxNext.length; i++) {
    idx[i].push(idxNext[i]);
  }
}

export function softmaxBatched(batches: Tensor2d) {
  return batches.map((batch) => softmax(batch));
}
