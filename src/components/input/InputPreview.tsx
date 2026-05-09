import { useState, useCallback, type ChangeEvent } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Upload, FileText, Globe } from 'lucide-react';
import { cn } from '@/lib/utils';

export type InputSource = 'war_and_peace' | 'shakespear' | 'upload';

export interface InputPreviewProps {
  selectedSource: InputSource | undefined;
  onSourceChange: (source: InputSource) => void;
  fileContent: string;
  fileName: string;
  onContentLoad: (content: string, fileName: string) => void;
}

export function InputPreview({
  selectedSource,
  onSourceChange,
  fileContent,
  fileName,
  onContentLoad,
}: InputPreviewProps) {
  const [isLoading, setIsLoading] = useState(false);

  const handleFileUpload = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      if (!file.name.endsWith('.txt')) {
        alert('Please select a .txt file');
        return;
      }

      setIsLoading(true);
      try {
        const content = await file.text();
        onContentLoad(content, file.name);
        onSourceChange('upload');
      } catch (error) {
        console.error('Failed to read file:', error);
        alert('Failed to read file');
      } finally {
        setIsLoading(false);
      }
    },
    [onContentLoad, onSourceChange],
  );

  const handleDefaultFileLoad = useCallback(
    async (source: 'war_and_peace' | 'shakespear') => {
      setIsLoading(true);
      try {
        let content: string;
        let name: string;

        if (source === 'war_and_peace') {
          const { default: warAndPeace } = await import('../../war_and_peace.txt?raw');
          content = warAndPeace;
          name = 'war_and_peace.txt';
        } else {
          const { default: shakespear } = await import('../../shakespear.txt?raw');
          content = shakespear;
          name = 'shakespear.txt';
        }

        onContentLoad(content, name);
        onSourceChange(source);
      } catch (error) {
        console.error('Failed to load default file:', error);
        alert('Failed to load default file');
      } finally {
        setIsLoading(false);
      }
    },
    [onContentLoad, onSourceChange],
  );

  const setInputSource = useCallback(
    (source: InputSource) => {
      if (source !== 'upload') {
        void handleDefaultFileLoad(source);
      } else {
        onSourceChange(source);
      }
    },
    [onSourceChange, handleDefaultFileLoad],
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle>Input Source Selection</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-3">
          <Label className="text-sm font-medium">Choose input source:</Label>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
            <Button
              variant={selectedSource === 'war_and_peace' ? 'default' : 'outline'}
              onClick={() => handleDefaultFileLoad('war_and_peace')}
              disabled={isLoading}
              className={cn(
                'flex items-center gap-2 h-auto p-3 text-left justify-start',
                selectedSource === 'war_and_peace' && 'ring-2 ring-primary',
              )}
            >
              <FileText className="size-4" />
              <div className="flex flex-col">
                <span className="font-medium">war_and_peace.txt</span>
                <span className="text-xs opacity-70">Russian/French text</span>
              </div>
            </Button>

            <Button
              variant={selectedSource === 'shakespear' ? 'default' : 'outline'}
              onClick={() => handleDefaultFileLoad('shakespear')}
              disabled={isLoading}
              className={cn(
                'flex items-center gap-2 h-auto p-3 text-left justify-start',
                selectedSource === 'shakespear' && 'ring-2 ring-primary',
              )}
            >
              <FileText className="size-4" />
              <div className="flex flex-col">
                <span className="font-medium">shakespear.txt</span>
                <span className="text-xs opacity-70">English text</span>
              </div>
            </Button>

            <Button
              variant={selectedSource === 'upload' ? 'default' : 'outline'}
              onClick={() => setInputSource('upload')}
              disabled={isLoading}
              className={cn(
                'flex items-center gap-2 h-auto p-3 text-left justify-start',
                selectedSource === 'upload' && 'ring-2 ring-primary',
              )}
            >
              <Upload className="size-4" />
              <div className="flex flex-col">
                <span className="font-medium">Upload file</span>
                <span className="text-xs opacity-70">Custom .txt</span>
              </div>
            </Button>
          </div>
        </div>

        {selectedSource === 'upload' && (
          <div className="space-y-2 p-4 border-2 border-dashed border-muted-foreground/25 rounded-lg">
            <Label htmlFor="file-upload" className="text-sm font-medium">
              Select .txt file:
            </Label>
            <Input
              id="file-upload"
              type="file"
              accept=".txt"
              onChange={handleFileUpload}
              disabled={isLoading}
              className="cursor-pointer"
            />
          </div>
        )}

        {fileContent && (
          <>
            <div className="text-sm font-medium">Current file: {fileName}</div>
            <div className="bg-muted p-3 rounded-md">
              <div className="text-xs text-muted-foreground mb-1">Preview (first 200 characters):</div>
              <pre className="text-sm whitespace-pre-wrap overflow-hidden">{fileContent.slice(0, 200)}...</pre>
            </div>
          </>
        )}

        {isLoading && <div className="text-sm text-muted-foreground">Loading file content...</div>}
      </CardContent>
    </Card>
  );
}
