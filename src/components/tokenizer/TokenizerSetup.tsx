import { ChevronLeft, ChevronRight, Loader } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';

import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

import type { Tokenizer } from '../../llm/types';
import type { TokenizerWorkerMessage, TokenizerWorkerResponse } from '../../workers/tokenizer.worker.ts';

import { BPETokenizer } from '../../llm/tokenizers/BPETokenizer';
import { CharTokenizer } from '../../llm/tokenizers/CharTokenizer';
import TokenizerWorker from '../../workers/tokenizer.worker.ts?worker';
import { TokenizerDemo } from './TokenizerDemo';
import { Vocabulary } from './Vocabulary';

export type TokenizerType = 'BPE' | 'Char';

export interface TokenizerSetupProps {
  fileContent: string;
  fileName: string;
  onComplete: (tokenizer: Tokenizer) => void;
  onBack: () => void;
}

type Step = 'setup' | 'introspect';

export function TokenizerSetup({ fileContent, fileName, onComplete, onBack }: TokenizerSetupProps) {
  const [currentStep, setCurrentStep] = useState<Step>('setup');
  const [tokenizerType, setTokenizerType] = useState<TokenizerType>('BPE');
  const [numMerges, setNumMerges] = useState(50);
  const [progress, setProgress] = useState(0);
  const [tokenizer, setTokenizer] = useState<Tokenizer | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isCreating, setIsCreating] = useState(false);

  const workerRef = useRef<Worker | null>(null);

  useEffect(() => {
    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
      }
    };
  }, []);

  const handleSelectType = useCallback((type: TokenizerType) => {
    setTokenizerType(type);
    // Reset state when changing tokenizer type
    setTokenizer(null);
    setProgress(0);
    setError(null);
    setIsCreating(false);
  }, []);

  const createTokenizer = useCallback(() => {
    if (!fileContent || isCreating) return;

    setIsCreating(true);
    setProgress(0);
    setError(null);
    setTokenizer(null);

    // Create and setup worker
    const worker = new TokenizerWorker();
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent<TokenizerWorkerResponse>) => {
      const { type, progress: workerProgress, tokenizer: tokenizerData, error: workerError } = event.data;

      if (type === 'progress') {
        setProgress(workerProgress || 0);
      } else if (type === 'complete') {
        setProgress(100);
        setIsCreating(false);

        // Recreate the tokenizer instance from worker data
        if (tokenizerData) {
          let newTokenizer: Tokenizer;
          if (tokenizerData.type === 'BPE') {
            newTokenizer = new BPETokenizer(tokenizerData.data.fileContent, tokenizerData.data.numMerges);
          } else {
            newTokenizer = new CharTokenizer(tokenizerData.data.fileContent);
          }
          setTokenizer(newTokenizer);
          setCurrentStep('introspect');
        }

        worker.terminate();
        workerRef.current = null;
      } else if (type === 'error') {
        setIsCreating(false);
        setError(workerError || 'Unknown error occurred');
        worker.terminate();
        workerRef.current = null;
      }
    };

    worker.onerror = () => {
      setIsCreating(false);
      setError('Worker error occurred');
      worker.terminate();
      workerRef.current = null;
    };

    // Send message to worker
    const message: TokenizerWorkerMessage = {
      type: 'create',
      tokenizerType,
      fileContent,
      numMerges: tokenizerType === 'BPE' ? numMerges : undefined,
    };

    worker.postMessage(message);
  }, [fileContent, tokenizerType, numMerges, isCreating]);

  const handleFinish = useCallback(() => {
    if (tokenizer) {
      onComplete(tokenizer);
    }
  }, [tokenizer, onComplete]);

  const handleBackToSetup = useCallback(() => {
    setCurrentStep('setup');
    setError(null);
  }, []);

  const renderSetupStep = () => (
    <Card className="p-6">
      <div className="space-y-6">
        <div>
          <h3 className="text-lg font-semibold">Create Tokenizer</h3>
          <p className="text-sm text-muted-foreground">Configure and create a tokenizer for "{fileName}"</p>
        </div>

        <div className="space-y-4">
          <div className="space-y-3">
            <Label htmlFor="tokenizer-select">Tokenizer Type</Label>
            <select
              className="
                w-full rounded-md border border-input bg-background p-2
                disabled:opacity-50
              "
              disabled={isCreating}
              id="tokenizer-select"
              onChange={(e) => {
                handleSelectType(e.target.value as TokenizerType);
              }}
              value={tokenizerType}
            >
              <option value="BPE">BPE Tokenizer (Byte Pair Encoding)</option>
              <option value="Char">Character Tokenizer</option>
            </select>

            <div className="text-sm text-muted-foreground">
              {tokenizerType === 'BPE' ? (
                <p>
                  <strong>BPE Tokenizer:</strong> Learns common character pairs to create a more efficient vocabulary.
                  Good for natural language and code. Requires configuration of merge operations.
                </p>
              ) : (
                <p>
                  <strong>Character Tokenizer:</strong> Simple character-level tokenization. Fast to create, larger
                  vocabulary, good for experimental purposes.
                </p>
              )}
            </div>
          </div>

          {tokenizerType === 'BPE' && (
            <div className="space-y-3">
              <Label htmlFor="num-merges">
                Number of Merges
                <Input
                  className="mt-1"
                  disabled={isCreating}
                  id="num-merges"
                  max="1000"
                  min="1"
                  onChange={(e) => {
                    setNumMerges(parseInt(e.target.value) || 50);
                  }}
                  type="number"
                  value={numMerges}
                />
              </Label>
              <div className="text-sm text-muted-foreground">
                Higher values create a more compact vocabulary but take longer to compute. Recommended: 50-200 for most
                texts.
              </div>
            </div>
          )}
        </div>

        {isCreating && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Loader className="size-4 animate-spin" />
              <span className="text-sm">
                {tokenizerType === 'BPE' ? 'Learning byte pair encodings...' : 'Processing character vocabulary...'} (
                {progress.toFixed(0)}%)
              </span>
            </div>

            <div className="h-2 w-full rounded-full bg-muted">
              <div
                className="
                  h-2 rounded-full bg-primary transition-all duration-300
                "
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {error && (
          <div
            className="
              rounded-md bg-destructive/10 p-3 text-sm text-destructive
            "
          >
            Error: {error}
          </div>
        )}

        <div className="flex justify-between">
          <Button disabled={isCreating} onClick={onBack} variant="outline">
            <ChevronLeft className="mr-1 size-4" />
            Back
          </Button>
          <Button disabled={isCreating} onClick={createTokenizer}>
            {isCreating ? (
              <>
                <Loader className="mr-2 size-4 animate-spin" />
                Creating...
              </>
            ) : (
              <>
                Create Tokenizer
                <ChevronRight className="ml-1 size-4" />
              </>
            )}
          </Button>
        </div>
      </div>
    </Card>
  );

  const renderIntrospectStep = () => (
    <div className="space-y-4">
      {tokenizer && (
        <>
          <Card className="p-6">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Tokenizer Overview</h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium">Type:</span> {tokenizerType} Tokenizer
                </div>
                <div>
                  <span className="font-medium">Vocabulary Size:</span> {tokenizer.getVocabSize()}
                </div>
                <div>
                  <span className="font-medium">Input File:</span> {fileName}
                </div>
                {tokenizerType === 'BPE' && (
                  <div>
                    <span className="font-medium">Merges:</span> {numMerges}
                  </div>
                )}
              </div>

              <div className="text-sm text-muted-foreground">
                {tokenizerType === 'BPE' ? (
                  <p>
                    <strong>BPE (Byte Pair Encoding):</strong> This tokenizer learned {numMerges} merge operations to
                    create a vocabulary of {tokenizer.getVocabSize()} tokens. It combines frequently occurring character
                    pairs to create more efficient token representations.
                  </p>
                ) : (
                  <p>
                    <strong>Character Tokenizer:</strong> This tokenizer uses individual characters as tokens, resulting
                    in a vocabulary of {tokenizer.getVocabSize()} unique characters from your input text.
                  </p>
                )}
              </div>
            </div>
          </Card>

          <Vocabulary tokenizer={tokenizer} />
          <TokenizerDemo tokenizer={tokenizer} />

          <div className="flex justify-between">
            <Button onClick={handleBackToSetup} variant="outline">
              <ChevronLeft className="mr-1 size-4" />
              Back: Change Tokenizer
            </Button>
            <Button onClick={handleFinish}>
              Next: Configure Model
              <ChevronRight className="ml-1 size-4" />
            </Button>
          </div>
        </>
      )}
    </div>
  );

  switch (currentStep) {
    case 'introspect':
      return renderIntrospectStep();
    case 'setup':
      return renderSetupStep();
    default:
      return null;
  }
}
