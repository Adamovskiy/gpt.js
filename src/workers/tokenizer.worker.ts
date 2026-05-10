import type { Tokenizer } from '../llm/types.ts';

import { BPETokenizer, type BPETokenizerSerializedData } from '../llm/tokenizers/BPETokenizer.ts';
import { CharTokenizer, type CharTokenizerSerializedData } from '../llm/tokenizers/CharTokenizer.ts';

export type TokenizerWorkerMessage = {
  fileContent: string;
  type: 'create';
} & (
  | {
      numMerges: number;
      tokenizerType: 'BPE';
    }
  | {
      tokenizerType: 'Char';
    }
);

export type TokenizerWorkerResponse =
  | {
      progress: number;
      type: 'progress';
    }
  | {
      error: string;
      type: 'error';
    }
  | {
      tokenizer:
        | {
            serializedData: BPETokenizerSerializedData;
            type: 'BPE';
          }
        | {
            serializedData: CharTokenizerSerializedData;
            type: 'Char';
          };
      type: 'complete';
    };

self.onmessage = (event: MessageEvent<TokenizerWorkerMessage>) => {
  const createEvent = event.data;

  try {
    let tokenizer: Tokenizer;

    if (createEvent.tokenizerType === 'BPE') {
      // Create BPE tokenizer with progress updates
      self.postMessage({ type: 'progress', progress: 0 });

      tokenizer = new BPETokenizer(createEvent.fileContent, createEvent.numMerges, (progress: number) => {
        self.postMessage({ type: 'progress', progress: Math.floor(progress * 100) });
      });

      self.postMessage({ type: 'progress', progress: 100 });
    } else {
      // CharTokenizer is fast, no need for progress
      self.postMessage({ type: 'progress', progress: 50 });
      tokenizer = new CharTokenizer(createEvent.fileContent);
      self.postMessage({ type: 'progress', progress: 100 });
    }

    // Send the completed tokenizer data
    self.postMessage({
      type: 'complete',
      tokenizer: {
        type: createEvent.tokenizerType,
        serializedData: tokenizer.getSerializedData(),
      },
    });
  } catch (error) {
    self.postMessage({
      type: 'error',
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
};
