import { ChevronLeft } from 'lucide-react';
import { type Dispatch, type SetStateAction } from 'react';

import type { SelectedFile } from '@/components/input/InputConfig.tsx';
import type { Optimizer } from '@/llm/optimizers/utils.ts';
import type { LanguageModel, Tokenizer } from '@/llm/types.ts';

import { ModelInference } from '@/components/train/ModelInference.tsx';
import { Button } from '@/components/ui/button.tsx';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.tsx';

import { ModelTraining } from './ModelTraining';

export function ModelUsage({
  tokenizer,
  model,
  onBack,
  selectedFile,
  optimizer,
  lossChartData,
  setLossChartData,
  setModel,
  setOptimizer,
}: {
  lossChartData: number[];
  model: LanguageModel;
  onBack: () => void;
  optimizer: Optimizer;
  selectedFile: SelectedFile;
  setLossChartData: Dispatch<SetStateAction<number[]>>;
  setModel: Dispatch<SetStateAction<LanguageModel | null>>;
  setOptimizer: Dispatch<SetStateAction<Optimizer | null | undefined>>;
  tokenizer: Tokenizer;
}) {
  return (
    <>
      <div className="mb-4 flex justify-between">
        <Button onClick={onBack} variant="outline">
          <ChevronLeft className="mr-1 size-4" />
          Back: Change Optimizer
        </Button>
      </div>

      <Tabs className="w-full" defaultValue="train">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="train">Training</TabsTrigger>
          <TabsTrigger value="generate">Inference</TabsTrigger>
        </TabsList>

        <TabsContent className="space-y-6" value="train">
          <ModelTraining
            lossChartData={lossChartData}
            model={model}
            optimizer={optimizer}
            selectedFile={selectedFile}
            setLossChartData={setLossChartData}
            setModel={setModel}
            setOptimizer={setOptimizer}
            tokenizer={tokenizer}
          />
        </TabsContent>

        <TabsContent className="space-y-6" value="generate">
          <ModelInference model={model} tokenizer={tokenizer} />
        </TabsContent>
      </Tabs>
    </>
  );
}
