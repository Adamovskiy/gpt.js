import { readFileSync } from 'fs';
import { CharTokenizer } from './tokenizer.js';
import { seed } from './random.js';
import { BigramLanguageModel } from './tfOps.js';
import { getBatch } from './sampling.js';
import {
  lowerTriangularMatrixAvgWeighted,
  lowerTriangularMatrixAvgWeightedSoftmax,
  getBagOfWordsOptimized,
  getBagOfWordsUnoptimized,
  matrixMultiply,
} from './tensorOps.js';
import { AdamWOptimizer, type Optimizer, SDGOptimizer } from './optimizers.js';

seed(42);

const fileContent = readFileSync('./src/voynaimir.txt', 'utf8');
const tokenizer = new CharTokenizer(fileContent);
const numberEmbeddingDimensions = 32;

const data = tokenizer.encode(fileContent);
const splitIndex = 0.9 * data.length;
const trainData = data.slice(0, splitIndex);
const validationData = data.slice(splitIndex);

const model = new BigramLanguageModel(tokenizer.getVocabSize(), numberEmbeddingDimensions);

// Learning loop
let loss;
for (let i = 0; i < 100000; i++) {
  const { contexts, outputs } = getBatch(trainData);
  // const optimizer: Optimizer = new SDGOptimizer(model, 1e-3);
  const optimizer: Optimizer = new AdamWOptimizer(model, 1e-3, 0.9, 0.999, 1e-8, 0.01);
  loss = optimizer.train(contexts, outputs);
  console.log(`Loss: ${loss} (perfect - 0, random - ${-Math.log(1 / tokenizer.getVocabSize())})`);
}

const output = model.generate([[42]], 100);
console.log(tokenizer.decode(output[0]));

process.exit();

const tri = lowerTriangularMatrixAvgWeighted(3);
console.log(tri);

console.log(lowerTriangularMatrixAvgWeightedSoftmax(3));

console.log(
  matrixMultiply(tri, [
    [2, 7],
    [6, 4],
    [6, 5],
  ]),
);

console.log(
  getBagOfWordsUnoptimized([
    [
      [2, 7],
      [6, 4],
      [6, 5],
    ],
  ]),
);

console.log(
  getBagOfWordsOptimized([
    [
      [2, 7],
      [6, 4],
      [6, 5],
    ],
  ]),
);
