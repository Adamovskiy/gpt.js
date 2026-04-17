import {
  lowerTriangularMatrixAvgWeighted,
  lowerTriangularMatrixAvgWeightedSoftmax,
  matrixMultiply,
  getBagOfWordsOptimized,
  getBagOfWordsUnoptimized,
} from './tensorOps.js';

const ITERATIONS = 10_000;

function bench(name: string, fn: () => void, iterations = ITERATIONS): void {
  // Warmup
  for (let i = 0; i < Math.ceil(iterations / 10); i++) fn();

  const start = performance.now();
  for (let i = 0; i < iterations; i++) fn();
  const elapsed = performance.now() - start;

  const avgUs = (elapsed / iterations) * 1000;
  console.log(`  ${name.padEnd(52)} avg ${avgUs.toFixed(3).padStart(8)} µs  (${iterations} iters)`);
}

// --- Test data ---------------------------------------------------------------

const TRI_SIZE = 32;
const SEQ_LEN = 32;
const EMBED_DIM = 64;
const BATCH_SIZE = 4;

const tri3 = lowerTriangularMatrixAvgWeighted(3);
const embeddings3x2 = [
  [2, 7],
  [6, 4],
  [6, 5],
];

const triLarge = lowerTriangularMatrixAvgWeighted(TRI_SIZE);
const embeddingsLarge = Array.from({ length: TRI_SIZE }, () => Array.from({ length: EMBED_DIM }, () => Math.random()));

const bagInput: number[][][] = Array.from({ length: BATCH_SIZE }, () =>
  Array.from({ length: SEQ_LEN }, () => Array.from({ length: EMBED_DIM }, () => Math.random())),
);

// --- Benchmarks --------------------------------------------------------------

console.log('\nBenchmark results:\n');

console.log('lowerTriangularMatrixAvgWeighted:');
bench('size=3', () => lowerTriangularMatrixAvgWeighted(3));
bench(`size=${TRI_SIZE}`, () => lowerTriangularMatrixAvgWeighted(TRI_SIZE));

console.log('\nlowerTriangularMatrixAvgWeightedSoftmax:');
bench('size=3', () => lowerTriangularMatrixAvgWeightedSoftmax(3));
bench(`size=${TRI_SIZE}`, () => lowerTriangularMatrixAvgWeightedSoftmax(TRI_SIZE));

console.log('\nmatrixMultiply:');
bench('tri(3) × 3×2', () => matrixMultiply(tri3, embeddings3x2));
bench(`tri(${TRI_SIZE}) × ${TRI_SIZE}×${EMBED_DIM}`, () => matrixMultiply(triLarge, embeddingsLarge));

console.log('\ngetBagOfWordsUnoptimized:');
bench(`batch=${BATCH_SIZE}, seq=${SEQ_LEN}, dim=${EMBED_DIM}`, () => getBagOfWordsUnoptimized(bagInput));

console.log('\ngetBagOfWordsOptimized:');
bench(`batch=${BATCH_SIZE}, seq=${SEQ_LEN}, dim=${EMBED_DIM}`, () => getBagOfWordsOptimized(bagInput));

console.log('\nOptimized vs Unoptimized speedup:');
const iterSmall = 1_000;
const t0 = performance.now();
for (let i = 0; i < iterSmall; i++) getBagOfWordsUnoptimized(bagInput);
const tUnopt = performance.now() - t0;

const t1 = performance.now();
for (let i = 0; i < iterSmall; i++) getBagOfWordsOptimized(bagInput);
const tOpt = performance.now() - t1;

const speedup = tUnopt / tOpt;
console.log(
  `  Unoptimized: ${(tUnopt / iterSmall).toFixed(3)} ms/iter   Optimized: ${(tOpt / iterSmall).toFixed(3)} ms/iter   speedup: ${speedup.toFixed(2)}x`,
);
