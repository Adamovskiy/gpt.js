import { readFileSync } from 'fs';
import { CharTokenizer } from './tokenizer.js';
import { seed } from './random.js';
import { GPTModel } from './tfOps.js';
import { blockSize, getBatch } from './sampling.js';
import { UniversalAdamWOptimizer } from './optimizers.js';

seed(42);

const fileContent = readFileSync('./src/voynaimir.txt', 'utf8');
const tokenizer = new CharTokenizer(fileContent);
const numberEmbeddingDimensions = 32;

const data = tokenizer.encode(fileContent);
const splitIndex = 0.9 * data.length;
const trainData = data.slice(0, splitIndex);

const numHeads = 4;
const numLayers = 3;
const model = new GPTModel(tokenizer.getVocabSize(), numberEmbeddingDimensions, blockSize, numHeads, numLayers);

// Learning loop
const optimizer = new UniversalAdamWOptimizer(model, 1e-3, 0.9, 0.999, 1e-8, 0.01);
let loss;
for (let i = 0; i < 10000; i++) {
  const { contexts, outputs } = getBatch(trainData);
  loss = optimizer.train(contexts, outputs);
  console.log(`Loss: ${loss} (perfect - 0, random - ${-Math.log(1 / tokenizer.getVocabSize())})`);
}

const output = model.generate([[42]], 100);
console.log(tokenizer.decode(output[0]));
