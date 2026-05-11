import { useMemo, useState } from 'react';

import type { Tokenizer } from '@/llm/types.ts';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';

function parseTokenInput(input: string): {
  error: string | null;
  tokens: number[];
} {
  const trimmed = input.trim();

  if (!trimmed) {
    return { tokens: [], error: null };
  }

  const parts = trimmed
    .replace(/[[\]]/g, '')
    .split(/[\s,]+/)
    .filter(Boolean);

  const tokens = parts.map(Number);

  const invalidIndex = tokens.findIndex((value) => !Number.isInteger(value));

  if (invalidIndex !== -1) {
    return {
      tokens: [],
      error: `Invalid token: "${parts[invalidIndex]}"`,
    };
  }

  return { tokens, error: null };
}

function getTokenColors(count: number): string[] {
  const colors = [
    'bg-red-100 text-red-800 border-red-200',
    'bg-blue-100 text-blue-800 border-blue-200',
    'bg-green-100 text-green-800 border-green-200',
    'bg-yellow-100 text-yellow-800 border-yellow-200',
    'bg-purple-100 text-purple-800 border-purple-200',
    'bg-pink-100 text-pink-800 border-pink-200',
    'bg-indigo-100 text-indigo-800 border-indigo-200',
    'bg-orange-100 text-orange-800 border-orange-200',
    'bg-teal-100 text-teal-800 border-teal-200',
    'bg-cyan-100 text-cyan-800 border-cyan-200',
  ];

  return Array.from({ length: count }, (_, i) => colors[i % colors.length]);
}

export function TokenizerDemo({ tokenizer, inputContent }: { inputContent: string; tokenizer: Tokenizer }) {
  const [text, setText] = useState('');
  const [tokenInput, setTokenInput] = useState('');

  const encodedTokens = useMemo(() => {
    return tokenizer.encode(text);
  }, [tokenizer, text]);

  const parsedTokenInput = useMemo(() => {
    return parseTokenInput(tokenInput);
  }, [tokenInput]);

  const decodedText = useMemo(() => {
    if (parsedTokenInput.error) return '';
    return tokenizer.decode(parsedTokenInput.tokens);
  }, [tokenizer, parsedTokenInput]);

  const visualizationData = useMemo(() => {
    const textToVisualize = inputContent.slice(0, 200);
    const tokens = tokenizer.encode(textToVisualize);
    const colors = getTokenColors(tokens.length);

    const segments: { color: string; text: string; tokenId: number }[] = [];
    let currentPosition = 0;

    tokens.forEach((tokenId, index) => {
      const tokenText = tokenizer.decode([tokenId]);
      const tokenLength = tokenText.length;

      if (currentPosition + tokenLength <= textToVisualize.length) {
        segments.push({
          text: tokenText,
          color: colors[index],
          tokenId,
        });
        currentPosition += tokenLength;
      }
    });

    return { segments, totalTokens: tokens.length, textLength: textToVisualize.length };
  }, [tokenizer, inputContent]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Tokenizer demo</CardTitle>
        <CardDescription>Encode text into token ids or decode token ids back into text.</CardDescription>
      </CardHeader>

      <CardContent>
        <Tabs className="w-full" defaultValue="encode">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="encode">Encode</TabsTrigger>
            <TabsTrigger value="decode">Decode</TabsTrigger>
            <TabsTrigger value="visualize">Visualize</TabsTrigger>
          </TabsList>

          <TabsContent className="mt-4 space-y-4" value="encode">
            <div className="space-y-2">
              <Label htmlFor="tokenizer-text">Text</Label>
              <Textarea
                className="min-h-28 resize-y font-mono"
                id="tokenizer-text"
                onChange={(event) => {
                  setText(event.target.value);
                }}
                placeholder="Enter text to tokenize..."
                value={text}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <Label>tokenizer.encode(Text)</Label>
                <Badge variant="secondary">{encodedTokens.length} tokens</Badge>
              </div>

              <div className="min-h-16 rounded-md border bg-muted/30 p-3">
                {encodedTokens.length > 0 ? (
                  <div className="flex flex-wrap gap-1.5">
                    [
                    {encodedTokens.map((token, index) => (
                      <Badge className="font-mono" key={`${index}-${token}`} variant="outline">
                        {token}
                      </Badge>
                    ))}
                    ]
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No tokens.</p>
                )}
              </div>
            </div>
          </TabsContent>

          <TabsContent className="mt-4 space-y-4" value="decode">
            <div className="space-y-2">
              <Label htmlFor="tokenizer-tokens">Token ids</Label>
              <Textarea
                className="min-h-28 resize-y font-mono"
                id="tokenizer-tokens"
                onChange={(event) => {
                  setTokenInput(event.target.value);
                }}
                placeholder="Example: 12 45 78 or [12, 45, 78]"
                value={tokenInput}
              />
              {parsedTokenInput.error && <p className="text-sm text-destructive">{parsedTokenInput.error}</p>}
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <Label>tokenizer.decode(Token ids)</Label>
                <Badge variant="secondary">{parsedTokenInput.tokens.length} tokens</Badge>
              </div>

              <div
                className="
                  min-h-16 rounded-md border bg-muted/30 p-3 font-mono text-sm
                  whitespace-pre-wrap
                "
              >
                {decodedText || <span className="text-muted-foreground">No decoded text.</span>}
              </div>
            </div>
          </TabsContent>

          <TabsContent className="mt-4 space-y-4" value="visualize">
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <Label>Training data tokenization</Label>
                <div className="flex items-center gap-2">
                  <Badge variant="secondary">{visualizationData.totalTokens} tokens</Badge>
                  <Badge variant="outline">{visualizationData.textLength} chars</Badge>
                </div>
              </div>

              <div className="min-h-16 rounded-md border bg-muted/30 p-3">
                {visualizationData.segments.length > 0 ? (
                  <div className="font-mono text-sm/relaxed">
                    {visualizationData.segments.map((segment, index) => (
                      <span
                        className={`inline-block rounded-sm border px-1 py-0.5 ${segment.color} mr-0.5 mb-0.5`}
                        key={`${index}-${segment.tokenId}`}
                        title={`Token ID: ${segment.tokenId}`}
                      >
                        {segment.text}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No content to visualize.</p>
                )}
              </div>

              {inputContent.length > 200 && (
                <p className="text-xs text-muted-foreground">
                  Showing first 200 characters of {inputContent.length} total characters.
                </p>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
