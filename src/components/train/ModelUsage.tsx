import { type Dispatch, type SetStateAction } from 'react';

import type { SelectedFile } from '@/components/input/InputConfig.tsx';
import type { LanguageModel, Optimizer, Tokenizer } from '@/llm/types.ts';

import { BackButton } from '@/components/layout/BackButton.tsx';
import { StepLayout } from '@/components/layout/StepLayout.tsx';
import { ModelInference } from '@/components/train/ModelInference.tsx';
import { Card, CardContent } from '@/components/ui/card.tsx';
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
    <StepLayout
      backButton={<BackButton onClick={onBack} />}
      subtitle="Train your model and generate text with it"
      title="6. Use model"
    >
      <Card>
        <CardContent>
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
        </CardContent>
      </Card>
    </StepLayout>
  );
}
