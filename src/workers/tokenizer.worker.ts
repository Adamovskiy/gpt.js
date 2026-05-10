import { BPETokenizer } from '../llm/tokenizers/BPETokenizer.ts';
import { CharTokenizer } from '../llm/tokenizers/CharTokenizer.ts';
import type { Tokenizer } from '../llm/types.ts';

export interface TokenizerWorkerMessage {
  type: 'create';
  tokenizerType: 'BPE' | 'Char';
  fileContent: string;
  numMerges?: number;
}

export interface TokenizerWorkerResponse {
  type: 'progress' | 'complete' | 'error';
  progress?: number;
  tokenizer?: {
    type: string;
    vocab: string[];
    vocabSize: number;
    // We'll serialize the tokenizer data instead of the class instance
    data: { fileContent: string; numMerges?: number };
  };
  error?: string;
}

self.onmessage = async (event: MessageEvent<TokenizerWorkerMessage>) => {
  const { type, tokenizerType, fileContent, numMerges } = event.data;

  if (type === 'create') {
    try {
      let tokenizer: Tokenizer;

      if (tokenizerType === 'BPE') {
        // Create BPE tokenizer with progress updates
        self.postMessage({ type: 'progress', progress: 10 } as TokenizerWorkerResponse);

        tokenizer = new BPETokenizer(fileContent, numMerges || 50);

        self.postMessage({ type: 'progress', progress: 100 } as TokenizerWorkerResponse);
      } else {
        // CharTokenizer is fast, no need for progress
        self.postMessage({ type: 'progress', progress: 50 } as TokenizerWorkerResponse);
        tokenizer = new CharTokenizer(fileContent);
        self.postMessage({ type: 'progress', progress: 100 } as TokenizerWorkerResponse);
      }

      // Send the completed tokenizer data
      self.postMessage({
        type: 'complete',
        tokenizer: {
          type: tokenizerType,
          vocab: tokenizer.getVocab(),
          vocabSize: tokenizer.getVocabSize(),
          data:
            tokenizerType === 'BPE'
              ? {
                  fileContent,
                  numMerges: numMerges || 50,
                }
              : {
                  fileContent,
                },
        },
      } as TokenizerWorkerResponse);
    } catch (error) {
      self.postMessage({
        type: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
      } as TokenizerWorkerResponse);
    }
  }
};
