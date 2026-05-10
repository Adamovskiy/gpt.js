import { type ChangeEvent, useCallback, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { FileText, Upload } from 'lucide-react';
import { cn } from '@/lib/utils';
import { InputPreview } from './InputPreview.tsx';

type InputSource = 'war_and_peace' | 'shakespear' | 'upload';

export type SelectedFile = {
  content: string;
  name: string;
};

export function InputConfig({
  selectedFile,
  onSelectedFileChange,
}: {
  selectedFile: SelectedFile | undefined;
  onSelectedFileChange: (selectedFile: SelectedFile) => void;
}) {
  const [inputSource, setInputSource] = useState<InputSource>();

  const [isLoading, setIsLoading] = useState(false);

  const handleCustomFileUpload = useCallback(
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
        onSelectedFileChange({
          content,
          name: file.name,
        });
        setInputSource('upload');
      } catch (error) {
        console.error('Failed to read file:', error);
        alert('Failed to read file');
      } finally {
        setIsLoading(false);
      }
    },
    [onSelectedFileChange],
  );

  const handlePresetFileUpload = useCallback(
    async (source: 'war_and_peace' | 'shakespear') => {
      setIsLoading(true);
      try {
        let content: string;
        let name: string;

        if (source === 'war_and_peace') {
          const { default: warAndPeace } = await import('../../data/war_and_peace.txt?raw');
          content = warAndPeace;
          name = 'war_and_peace.txt';
        } else {
          const { default: shakespear } = await import('../../data/shakespear.txt?raw');
          content = shakespear;
          name = 'shakespear.txt';
        }

        onSelectedFileChange({ content, name });
        setInputSource(source);
      } catch (error) {
        console.error('Failed to load default file:', error);
        alert('Failed to load default file');
      } finally {
        setIsLoading(false);
      }
    },
    [onSelectedFileChange],
  );

  const selectInputSource = useCallback(
    (source: InputSource) => {
      if (source !== 'upload') {
        void handlePresetFileUpload(source);
      } else {
        setInputSource(source);
      }
    },
    [handlePresetFileUpload],
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
              variant={inputSource === 'war_and_peace' ? 'default' : 'outline'}
              onClick={() => handlePresetFileUpload('war_and_peace')}
              disabled={isLoading}
              className={cn(
                'flex items-center gap-2 h-auto p-3 text-left justify-start',
                inputSource === 'war_and_peace' && 'ring-2 ring-primary',
              )}
            >
              <FileText className="size-4" />
              <div className="flex flex-col">
                <span className="font-medium">war_and_peace.txt</span>
                <span className="text-xs opacity-70">Russian/French text</span>
              </div>
            </Button>

            <Button
              variant={inputSource === 'shakespear' ? 'default' : 'outline'}
              onClick={() => handlePresetFileUpload('shakespear')}
              disabled={isLoading}
              className={cn(
                'flex items-center gap-2 h-auto p-3 text-left justify-start',
                inputSource === 'shakespear' && 'ring-2 ring-primary',
              )}
            >
              <FileText className="size-4" />
              <div className="flex flex-col">
                <span className="font-medium">shakespear.txt</span>
                <span className="text-xs opacity-70">English text</span>
              </div>
            </Button>

            <Button
              variant={inputSource === 'upload' ? 'default' : 'outline'}
              onClick={() => selectInputSource('upload')}
              disabled={isLoading}
              className={cn(
                'flex items-center gap-2 h-auto p-3 text-left justify-start',
                inputSource === 'upload' && 'ring-2 ring-primary',
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

        {inputSource === 'upload' && (
          <div className="space-y-2 p-4 border-2 border-dashed border-muted-foreground/25 rounded-lg">
            <Label htmlFor="file-upload" className="text-sm font-medium">
              Select .txt file:
            </Label>
            <Input
              id="file-upload"
              type="file"
              accept=".txt"
              onChange={handleCustomFileUpload}
              disabled={isLoading}
              className="cursor-pointer"
            />
          </div>
        )}

        {selectedFile && <InputPreview fileName={selectedFile.name} fileContent={selectedFile.content} />}

        {isLoading && <div className="text-sm text-muted-foreground">Loading file content...</div>}
      </CardContent>
    </Card>
  );
}
