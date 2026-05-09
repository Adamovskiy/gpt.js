import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';
import { useMemo, useState } from 'react';
import type { Tokenizer } from '../../llm/types.ts';

function parseTokenInput(input: string): {
  tokens: number[];
  error: string | null;
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

export function TokenizerDemo({ tokenizer }: { tokenizer: Tokenizer }) {
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

  return (
    <Card>
      <CardHeader>
        <CardTitle>Tokenizer demo</CardTitle>
        <CardDescription>Encode text into token ids or decode token ids back into text.</CardDescription>
      </CardHeader>

      <CardContent>
        <Tabs defaultValue="encode" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="encode">Encode</TabsTrigger>
            <TabsTrigger value="decode">Decode</TabsTrigger>
          </TabsList>

          <TabsContent value="encode" className="mt-4 space-y-4">
            <div className="space-y-2">
              <Label htmlFor="tokenizer-text">Text</Label>
              <Textarea
                id="tokenizer-text"
                value={text}
                onChange={(event) => setText(event.target.value)}
                placeholder="Enter text to tokenize..."
                className="min-h-28 resize-y font-mono"
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
                      <Badge key={`${index}-${token}`} variant="outline" className="font-mono">
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

          <TabsContent value="decode" className="mt-4 space-y-4">
            <div className="space-y-2">
              <Label htmlFor="tokenizer-tokens">Token ids</Label>
              <Textarea
                id="tokenizer-tokens"
                value={tokenInput}
                onChange={(event) => setTokenInput(event.target.value)}
                placeholder="Example: 12 45 78 or [12, 45, 78]"
                className="min-h-28 resize-y font-mono"
              />
              {parsedTokenInput.error && <p className="text-sm text-destructive">{parsedTokenInput.error}</p>}
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <Label>tokenizer.decode(Token ids)</Label>
                <Badge variant="secondary">{parsedTokenInput.tokens.length} tokens</Badge>
              </div>

              <div className="min-h-16 whitespace-pre-wrap rounded-md border bg-muted/30 p-3 font-mono text-sm">
                {decodedText || <span className="text-muted-foreground">No decoded text.</span>}
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
