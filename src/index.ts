import { readFileSync } from 'fs';
import { CharTokenizer } from './tokenizer.js';
import { seed } from './random.js';
import { BigramLanguageModel } from './tfOps.js';
import { getBatch } from './sampling.js';

seed(42);

const fileContent = readFileSync('./src/voynaimir.txt', 'utf8');
const tokenizer = new CharTokenizer(fileContent);

const data = tokenizer.encode(fileContent);
const splitIndex = 0.9 * data.length;
const trainData = data.slice(0, splitIndex);
const validationData = data.slice(splitIndex);

const { contexts, outputs } = getBatch(trainData);
const model = new BigramLanguageModel(tokenizer.getVocabSize());
const { logits, loss } = model.forward(contexts, outputs);

console.log(`Loss: ${loss} (perfect - 0, random - ${-Math.log(1 / tokenizer.getVocabSize())})`);

const output = model.generate([[42]], 100);
console.log(tokenizer.decode(output[0]));
