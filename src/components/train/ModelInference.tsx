import { Loader } from 'lucide-react';
import { useCallback, useState } from 'react';

import type { LanguageModel, Tokenizer } from '@/llm/types.ts';

import { Button } from '@/components/ui/button.tsx';
import { Input } from '@/components/ui/input.tsx';
import { Label } from '@/components/ui/label.tsx';

export function ModelInference({ tokenizer, model }: { model: LanguageModel; tokenizer: Tokenizer }) {
  const [initialString, setInitialString] = useState('');
  const [maxNewTokens, setMaxNewTokens] = useState(100);
  const [generateOutput, setGenerateOutput] = useState<string>();
  const [generateInProgress, setGenerateInProgress] = useState<boolean>(false);

  const generate = useCallback(async () => {
    try {
      setGenerateInProgress(true);
      const initialTokens = tokenizer.encode(initialString);
      const output = await model.generate([initialTokens], maxNewTokens);
      setGenerateOutput(tokenizer.decode(output[0]));
    } catch (error) {
      console.error('Generation failed:', error);
      setGenerateOutput('Error: Generation failed');
    } finally {
      setGenerateInProgress(false);
    }
  }, [tokenizer, model, initialString, maxNewTokens]);

  return (
    <>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Label>
          Initial Context
          <Input
            className="mt-1"
            onChange={(e) => {
              setInitialString(e.target.value);
            }}
            placeholder="Enter some text to start generation..."
            value={initialString}
          />
        </Label>
        <Label>
          Max New Tokens
          <Input
            className="mt-1"
            onChange={(e) => {
              setMaxNewTokens(parseInt(e.target.value));
            }}
            type={'number'}
            value={maxNewTokens}
          />
        </Label>
      </div>

      <div className="flex items-center gap-4">
        <Button disabled={generateInProgress} onClick={() => void generate()} variant="outline">
          {generateInProgress && <Loader className="mr-2 animate-spin" />}
          {generateInProgress ? 'Generating...' : 'Generate Text'}
        </Button>
      </div>

      {generateOutput && (
        <div className="rounded-lg border p-4">
          <div className="mb-2 text-sm font-medium">Generated Output:</div>
          <code
            className="
                      block overflow-auto rounded-sm bg-muted p-2 text-xs
                      whitespace-pre-wrap
                    "
          >
            {generateOutput}
          </code>
        </div>
      )}
    </>
  );
}
