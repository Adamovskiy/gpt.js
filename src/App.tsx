import { useCallback, useEffect, useState } from 'react';

import type { LanguageModel, Tokenizer } from '@/llm/types.ts';

import { InputConfig, type SelectedFile } from '@/components/input/InputConfig.tsx';
import { ModelConfig } from '@/components/model/ModelConfig.tsx';
import { OptimizerConfig } from '@/components/optimizer/OptimizerConfig.tsx';
import { TokenizerSetup } from '@/components/tokenizer/TokenizerSetup.tsx';
import { ModelUsage } from '@/components/train/ModelUsage.tsx';
import { Button } from '@/components/ui/button.tsx';
import { seed } from '@/lib/random.ts';
import { type Optimizer } from '@/llm/optimizers/utils.ts';

type AppStep = 'input' | 'tokenizer' | 'model' | 'optimizer' | 'train';

function App() {
  const [currentStep, setCurrentStep] = useState<AppStep>('input');
  const [tokenizer, setTokenizer] = useState<Tokenizer | null>(null);
  const [model, setModel] = useState<LanguageModel | null>(null);
  const [optimizer, setOptimizer] = useState<Optimizer | null>();
  const [lossChartData, setLossChartData] = useState<number[]>([]);
  const [selectedFile, setSelectedFile] = useState<SelectedFile | null>(null);

  useEffect(() => {
    // Init RNG on app mount with a fixed seed to ensure reproducible results
    seed(42);
  }, []);

  const handleTokenizerComplete = useCallback((newTokenizer: Tokenizer) => {
    setTokenizer(newTokenizer);
    setCurrentStep('model');
  }, []);

  const handleBackToInput = useCallback(() => {
    setCurrentStep('input');
    setTokenizer(null);
    setModel(null);
    setOptimizer(null);
    setLossChartData([]);
  }, []);

  const handleContentLoad = useCallback((selected: SelectedFile | null) => {
    setSelectedFile(selected);
    // Reset tokenizer when content changes
    setTokenizer(null);
    setModel(null);
    setOptimizer(null);
    setLossChartData([]);
    setCurrentStep('input');
  }, []);

  const handleProceedToTokenizer = useCallback(() => {
    if (selectedFile) {
      setCurrentStep('tokenizer');
    }
  }, [selectedFile]);

  const handleModelComplete = useCallback((newModel: LanguageModel | null) => {
    setModel(newModel);
    if (newModel) {
      setCurrentStep('optimizer');
    }
  }, []);

  const handleBackToModel = useCallback(() => {
    setCurrentStep('model');
  }, []);

  const handleOptimizerComplete = useCallback((newOptimizer: Optimizer) => {
    setOptimizer(newOptimizer);
    setCurrentStep('train');
  }, []);

  const handleBackToOptimizer = useCallback(() => {
    setCurrentStep('optimizer');
  }, []);

  const renderInputStep = () => (
    <div className="space-y-4">
      <InputConfig onSelectedFileChange={handleContentLoad} selectedFile={selectedFile} />
      {selectedFile && (
        <div className="mt-4">
          <Button onClick={handleProceedToTokenizer}>Next: Create Tokenizer</Button>
        </div>
      )}
    </div>
  );

  const renderTokenizerStep = () =>
    selectedFile && (
      <TokenizerSetup
        fileContent={selectedFile.content}
        fileName={selectedFile.name}
        onBack={handleBackToInput}
        onComplete={handleTokenizerComplete}
      />
    );

  const renderModelStep = () => (
    <div>
      {tokenizer && (
        <ModelConfig
          onBack={() => {
            setCurrentStep('tokenizer');
          }}
          onComplete={handleModelComplete}
          vocabSize={tokenizer.getVocabSize()}
        />
      )}
    </div>
  );

  const renderOptimizerStep = () => (
    <div>
      {model && <OptimizerConfig model={model} onBack={handleBackToModel} onComplete={handleOptimizerComplete} />}
    </div>
  );

  const renderTrainStep = () =>
    tokenizer &&
    model &&
    optimizer &&
    selectedFile && (
      <ModelUsage
        lossChartData={lossChartData}
        model={model}
        onBack={handleBackToOptimizer}
        optimizer={optimizer}
        selectedFile={selectedFile}
        setLossChartData={setLossChartData}
        setModel={setModel}
        setOptimizer={setOptimizer}
        tokenizer={tokenizer}
      />
    );

  return (
    <main className="mx-auto flex max-w-3xl flex-col gap-4 px-2 py-4">
      {currentStep === 'input' && renderInputStep()}
      {currentStep === 'tokenizer' && renderTokenizerStep()}
      {currentStep === 'model' && renderModelStep()}
      {currentStep === 'optimizer' && renderOptimizerStep()}
      {currentStep === 'train' && renderTrainStep()}
    </main>
  );
}

export default App;
