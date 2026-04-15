import { readFileSync } from 'fs';
import { CharTokenizer } from './tokenizer.js';

const trainingData = readFileSync('./src/voynaimir.txt', 'utf8');
const tokenizer = new CharTokenizer(trainingData);

console.log(tokenizer.encode('hello world'));
console.log(tokenizer.decode([59, 56, 59, 56]));
