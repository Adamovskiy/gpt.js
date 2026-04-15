import { readFileSync } from 'fs';
import { CharTokenizer } from './tokenizer.js';
import { random, seed } from './random.js';
import type { Tensor1d, Tensor2d } from './tensorOps.js';

seed(42);

const fileContent = readFileSync('./src/voynaimir.txt', 'utf8');
const tokenizer = new CharTokenizer(fileContent);

// console.log(tokenizer.encode('hello world'));
// console.log(tokenizer.decode([59, 56, 59, 56]));

const data = tokenizer.encode(fileContent);
const splitIndex = 0.9 * data.length;
const trainData = data.slice(0, splitIndex);
const validationData = data.slice(splitIndex);

const blockSize = 8; // Max context length for predictions
const batchSize = 4; // Independent sequences processed in parallel

function getBatch(data: Tensor1d): {
  contexts: Tensor2d;
  outputs: Tensor2d;
} {
  if (data.length < blockSize + 1) throw new Error(`Data is to small for blockSize ${blockSize}`);

  const contexts: Tensor2d = [];
  const outputs: Tensor2d = [];

  for (let batchIdx = 0; batchIdx < batchSize; batchIdx++) {
    const offset = Math.floor(random() * (data.length - blockSize));

    contexts.push(data.slice(offset, offset + blockSize));
    outputs.push(data.slice(offset + 1, offset + blockSize + 1));
  }

  // Two batchSize x blockSize 2D tensors
  return {
    contexts,
    outputs,
  };
}

console.log(getBatch(trainData));
