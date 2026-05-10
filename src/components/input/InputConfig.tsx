import { FileText, Upload } from 'lucide-react';
import { type ChangeEvent, useCallback, useState } from 'react';

import { cn } from '@/lib/utils';

import { Button } from '../ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { InputPreview } from './InputPreview.tsx';

type InputSource = 'war_and_peace' | 'shakespear' | 'upload';

export interface SelectedFile {
  content: string;
  name: string;
}

export function InputConfig({
  selectedFile,
  onSelectedFileChange,
}: {
  onSelectedFileChange: (selectedFile: SelectedFile) => void;
  selectedFile: SelectedFile | undefined;
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
          <div
            className="
              grid grid-cols-1 gap-2
              sm:grid-cols-3
            "
          >
            <Button
              className={cn(
                'flex h-auto items-center justify-start gap-2 p-3 text-left',
                inputSource === 'war_and_peace' && 'ring-2 ring-primary',
              )}
              disabled={isLoading}
              onClick={() => handlePresetFileUpload('war_and_peace')}
              variant={inputSource === 'war_and_peace' ? 'default' : 'outline'}
            >
              <FileText className="size-4" />
              <div className="flex flex-col">
                <span className="font-medium">war_and_peace.txt</span>
                <span className="text-xs opacity-70">Russian/French text</span>
              </div>
            </Button>

            <Button
              className={cn(
                'flex h-auto items-center justify-start gap-2 p-3 text-left',
                inputSource === 'shakespear' && 'ring-2 ring-primary',
              )}
              disabled={isLoading}
              onClick={() => handlePresetFileUpload('shakespear')}
              variant={inputSource === 'shakespear' ? 'default' : 'outline'}
            >
              <FileText className="size-4" />
              <div className="flex flex-col">
                <span className="font-medium">shakespear.txt</span>
                <span className="text-xs opacity-70">English text</span>
              </div>
            </Button>

            <Button
              className={cn(
                'flex h-auto items-center justify-start gap-2 p-3 text-left',
                inputSource === 'upload' && 'ring-2 ring-primary',
              )}
              disabled={isLoading}
              onClick={() => {
                selectInputSource('upload');
              }}
              variant={inputSource === 'upload' ? 'default' : 'outline'}
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
          <div
            className="
              space-y-2 rounded-lg border-2 border-dashed
              border-muted-foreground/25 p-4
            "
          >
            <Label className="text-sm font-medium" htmlFor="file-upload">
              Select .txt file:
            </Label>
            <Input
              accept=".txt"
              className="cursor-pointer"
              disabled={isLoading}
              id="file-upload"
              onChange={handleCustomFileUpload}
              type="file"
            />
          </div>
        )}

        {selectedFile && <InputPreview fileContent={selectedFile.content} fileName={selectedFile.name} />}

        {isLoading && <div className="text-sm text-muted-foreground">Loading file content...</div>}
      </CardContent>
    </Card>
  );
}
