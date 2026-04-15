import { readFileSync } from 'fs';

const file = readFileSync('./src/voynaimir.txt', 'utf8');
console.log(`File length: ${file.length}`);
