import assert from 'node:assert/strict';

const EPSILON = 1e-10;

export function assertClose(actual: number, expected: number, msg?: string) {
  assert.ok(Math.abs(actual - expected) < EPSILON, msg ?? `Expected ${actual} ≈ ${expected}`);
}

export function assertArraysClose(actual: number[], expected: number[], msg?: string) {
  assert.strictEqual(actual.length, expected.length, `${msg}: row count mismatch`);
  for (let i = 0; i < actual.length; i++) {
    assertClose(actual[i], expected[i], `${msg}: [${i}][${i}]`);
  }
}

export function assertMatrix2dClose(actual: number[][], expected: number[][], msg?: string) {
  assert.strictEqual(actual.length, expected.length, `${msg}: row count mismatch`);
  for (let i = 0; i < actual.length; i++) {
    assert.strictEqual(actual[i].length, expected[i].length, `${msg}: col count mismatch at row ${i}`);
    for (let j = 0; j < actual[i].length; j++) {
      assertClose(actual[i][j], expected[i][j], `${msg}: [${i}][${j}]`);
    }
  }
}

export function assertMatrix3dClose(actual: number[][][], expected: number[][][], msg?: string) {
  assert.strictEqual(actual.length, expected.length, `${msg}: batch count mismatch`);
  for (let b = 0; b < actual.length; b++) {
    assertMatrix2dClose(actual[b], expected[b], `${msg} batch=${b}`);
  }
}
