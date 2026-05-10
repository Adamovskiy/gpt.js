import assert from 'node:assert/strict';
import { it } from 'node:test';

import { assertArraysClose } from '../../testUtils.ts';
import { CharTokenizer } from './CharTokenizer.ts';

it('CharTokenizer', () => {
  const tokenizer = new CharTokenizer('aabbb!');
  // Each character is a token
  assert.strictEqual(tokenizer.getVocabSize(), 3);
  // Each token has its unique encoding number
  assertArraysClose(tokenizer.encode('baba'), [2, 1, 2, 1]);
  assert.strictEqual(tokenizer.decode([2, 2, 0, 0]), 'bb!!');
});
