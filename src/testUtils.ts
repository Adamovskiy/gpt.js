import { expect } from 'vitest';

const EPSILON = 1e-10;

export function assertClose(actual: number, expected: number, msg?: string) {
  expect(Math.abs(actual - expected) < EPSILON, msg ?? `Expected ${actual} ≈ ${expected}`).toBe(true);
}

export function assertArraysClose(actual: number[], expected: number[], msg?: string) {
  expect(actual.length, `${msg}: row count mismatch`).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    assertClose(actual[i], expected[i], `${msg}: [${i}][${i}]`);
  }
}

export function assertMatrix2dClose(actual: number[][], expected: number[][], msg?: string) {
  expect(actual.length, `${msg}: row count mismatch`).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    expect(actual[i].length, `${msg}: col count mismatch at row ${i}`).toBe(expected[i].length);
    for (let j = 0; j < actual[i].length; j++) {
      assertClose(actual[i][j], expected[i][j], `${msg}: [${i}][${j}]`);
    }
  }
}

export function assertMatrix3dClose(actual: number[][][], expected: number[][][], msg?: string) {
  expect(actual.length, `${msg}: batch count mismatch`).toBe(expected.length);
  for (let b = 0; b < actual.length; b++) {
    assertMatrix2dClose(actual[b], expected[b], `${msg} batch=${b}`);
  }
}
