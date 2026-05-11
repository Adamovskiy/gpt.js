import { Loader } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';

import type { Tokenizer } from '@/llm/types.ts';
import type { TokenizerWorkerMessage, TokenizerWorkerResponse } from '@/workers/tokenizer.worker.ts';

import { BackButton } from '@/components/layout/BackButton.tsx';
import { NextButton } from '@/components/layout/NextButton.tsx';
import { StepLayout } from '@/components/layout/StepLayout.tsx';
import { Card, CardContent } from '@/components/ui/card';
import { Field, FieldLabel } from '@/components/ui/field.tsx';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.tsx';
import { BPETokenizer } from '@/llm/tokenizers/BPETokenizer.ts';
import { CharTokenizer } from '@/llm/tokenizers/CharTokenizer.ts';

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

    const worker = new Worker(new URL('@/workers/tokenizer.worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent<TokenizerWorkerResponse>) => {
      const workerResponse = event.data;

      if (workerResponse.type === 'progress') {
        setProgress(workerResponse.progress);
      } else if (workerResponse.type === 'complete') {
        setProgress(100);
        setIsCreating(false);

        // Deserialize the tokenizer instance from worker data
        const tokenizer = workerResponse.tokenizer;
        let newTokenizer: Tokenizer;
        if (tokenizer.type === 'BPE') {
          newTokenizer = BPETokenizer.fromSerializedData(tokenizer.serializedData);
        } else {
          newTokenizer = CharTokenizer.fromSerializedData(tokenizer.serializedData);
        }
        setTokenizer(newTokenizer);
        setCurrentStep('introspect');

        worker.terminate();
        workerRef.current = null;
      } else {
        // if (workerResponse.type === 'error')
        setIsCreating(false);
        setError(workerResponse.error);
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
      fileContent,
      ...(tokenizerType === 'BPE' ? { tokenizerType, numMerges } : { tokenizerType }),
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
    <StepLayout
      backButton={<BackButton disabled={isCreating} onClick={onBack} />}
      completeButton={
        <NextButton
          disabled={isCreating}
          loading={isCreating}
          onClick={createTokenizer}
          title={isCreating ? 'Creating...' : 'Create tokenizer'}
        />
      }
      subtitle={`Configure and create a tokenizer for "${fileName}"`}
      title="2. Configure tokenizer"
    >
      <Card className="p-6">
        <div className="space-y-4">
          <div className="space-y-3">
            <Field>
              <FieldLabel>Tokenizer Type</FieldLabel>
              <Select
                disabled={isCreating}
                onValueChange={(value) => {
                  handleSelectType(value as TokenizerType);
                }}
                value={tokenizerType}
              >
                <SelectTrigger className="cursor-pointer">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="BPE">BPE Tokenizer (Byte Pair Encoding)</SelectItem>
                  <SelectItem value="Char">Character Tokenizer</SelectItem>
                </SelectContent>
              </Select>
            </Field>

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
                    const value = parseInt(e.target.value);
                    if (!isNaN(value)) {
                      setNumMerges(value);
                    }
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
      </Card>
    </StepLayout>
  );

  const renderIntrospectStep = () => (
    <StepLayout
      backButton={<BackButton onClick={handleBackToSetup} />}
      completeButton={<NextButton onClick={handleFinish} title="Configure model"></NextButton>}
      subtitle={
        'Review the details of your created tokenizer, explore its vocabulary, and see how it tokenizes your input text.'
      }
      title="3. Review tokenizer"
    >
      {tokenizer && (
        <>
          <Card>
            <CardContent>
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
            </CardContent>
          </Card>

          <Vocabulary tokenizer={tokenizer} />
          <TokenizerDemo inputContent={fileContent} tokenizer={tokenizer} />
        </>
      )}
    </StepLayout>
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
