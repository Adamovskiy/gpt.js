import { random } from '../../lib/random.ts';
import { matrixMultiply, sum1d, type Tensor1d } from '../tensorOps.ts';

export class Linear {
  readonly weights: number[][];
  readonly bias: number[];

  constructor(inputSize: number, outputSize: number, useBias = true) {
    this.weights = Array.from({ length: inputSize }, () => Array.from({ length: outputSize }, () => random() * 0.01));
    this.bias = useBias
      ? Array.from({ length: outputSize }, () => random() * 0.01)
      : new Array<number>(outputSize).fill(0);
  }

  // (inputSize) -> (outputSize)
  forward(input: Tensor1d): Tensor1d {
    return sum1d(matrixMultiply([input], this.weights)[0], this.bias);
  }
}
