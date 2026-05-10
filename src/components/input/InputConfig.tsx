import { FileText, Upload } from 'lucide-react';
import { useCallback, useState } from 'react';

import { InputPreview } from '@/components/input/InputPreview.tsx';
import { Button } from '@/components/ui/button.tsx';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { FileInput } from '@/components/ui/fileInput.tsx';
import { Label } from '@/components/ui/label';
import { cn } from '@/lib/utils';

type InputSource = 'war_and_peace' | 'shakespear' | 'upload';

export interface SelectedFile {
  content: string;
  name: string;
}

function InputOptionButton({
  variant,
  disabled,
  selected,
  onSelect,
  title,
  subtitle,
}: {
  disabled: boolean;
  onSelect: () => void;
  selected: boolean;
  subtitle: string;
  title: string;
  variant: 'upload' | 'preset';
}) {
  const Icon = variant === 'upload' ? Upload : FileText;
  return (
    <Button
      className={cn('flex h-auto items-center justify-start gap-2 p-3 text-left', selected && `ring-2 ring-primary`)}
      disabled={disabled}
      onClick={() => {
        onSelect();
      }}
      variant={selected ? 'default' : 'outline'}
    >
      <Icon className="size-4" />
      <div className="flex flex-col">
        <span className="font-medium">{title}</span>
        <span className="text-xs opacity-70">{subtitle}</span>
      </div>
    </Button>
  );
}

export function InputConfig({
  selectedFile,
  onSelectedFileChange,
}: {
  onSelectedFileChange: (selectedFile: SelectedFile | null) => void;
  selectedFile: SelectedFile | null;
}) {
  const [inputSource, setInputSource] = useState<InputSource>();

  const [isLoading, setIsLoading] = useState(false);

  const handleCustomFileUpload = useCallback(
    async (file: File | undefined) => {
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
      if (source === 'upload') {
        onSelectedFileChange(null);
        setInputSource(source);
      } else {
        void handlePresetFileUpload(source);
      }
    },
    [handlePresetFileUpload, onSelectedFileChange],
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
            <InputOptionButton
              disabled={isLoading}
              onSelect={() => {
                selectInputSource('war_and_peace');
              }}
              selected={inputSource === 'war_and_peace'}
              subtitle="A novel by Leo Tolstoy"
              title="war_and_peace.txt"
              variant="preset"
            />

            <InputOptionButton
              disabled={isLoading}
              onSelect={() => {
                selectInputSource('shakespear');
              }}
              selected={inputSource === 'shakespear'}
              subtitle="tiny_shakespeare dataset"
              title="shakespear.txt"
              variant="preset"
            />

            <InputOptionButton
              disabled={isLoading}
              onSelect={() => {
                selectInputSource('upload');
              }}
              selected={inputSource === 'upload'}
              subtitle="Custom .txt"
              title="Upload file"
              variant="upload"
            />
          </div>
        </div>

        {inputSource === 'upload' && (
          <FileInput
            accept=".txt"
            disabled={isLoading}
            label="Select .txt file:"
            onChange={(file) => {
              void handleCustomFileUpload(file);
            }}
          />
        )}

        {selectedFile && <InputPreview fileContent={selectedFile.content} fileName={selectedFile.name} />}

        {isLoading && <div className="text-sm text-muted-foreground">Loading file content...</div>}
      </CardContent>
    </Card>
  );
}
