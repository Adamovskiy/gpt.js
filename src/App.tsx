import { ChevronLeft, Loader } from 'lucide-react';
import { useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from 'recharts';

import { TokenizerSetup } from '@/components/tokenizer/TokenizerSetup.tsx';
import { Button } from '@/components/ui/button.tsx';

import type { LanguageModel, Tokenizer } from './llm/types.ts';

import { InputConfig, type SelectedFile } from './components/input/InputConfig.tsx';
import { ModelConfig } from './components/model/ModelConfig.tsx';
import { OptimizerConfig } from './components/optimizer/OptimizerConfig.tsx';
import { ChartContainer, ChartTooltip, ChartTooltipContent } from './components/ui/chart.tsx';
import { Input } from './components/ui/input.tsx';
import { Label } from './components/ui/label.tsx';
import { seed } from './lib/random.ts';
import { randomOutputLoss } from './llm/models/utils.ts';
import { type Optimizer } from './llm/optimizers/utils.ts';
import { getBatch } from './llm/sampling.ts';
import TrainWorker from './workers/train.worker.ts?worker';

type AppStep = 'input' | 'tokenizer' | 'model' | 'optimizer' | 'train';

function App() {
  const [currentStep, setCurrentStep] = useState<AppStep>('input');
  const [lossChartData, setLossChartData] = useState<number[]>([]);
  const [tokenizer, setTokenizer] = useState<Tokenizer | undefined>();
  const [model, setModel] = useState<LanguageModel | undefined>();
  const [optimizer, setOptimizer] = useState<Optimizer | undefined>();
  const [trainingInProgress, setTrainingInProgress] = useState(false);
  const [iterations, setIterations] = useState(100);
  const [initialString, setInitialString] = useState('');
  const [generateOutput, setGenerateOutput] = useState<string>();
  const [avgIterationTime, setAvgIterationTime] = useState<number | null>(null);
  const [fileContent, setFileContent] = useState<string>('');
  const [fileName, setFileName] = useState<string>('');

  useEffect(() => {
    // Init RNG on app mount
    seed(42);
  }, []);

  const handleTokenizerComplete = useCallback((newTokenizer: Tokenizer) => {
    setTokenizer(newTokenizer);
    setCurrentStep('model');
  }, []);

  const handleBackToInput = useCallback(() => {
    setCurrentStep('input');
    setTokenizer(undefined);
    setModel(undefined);
    setOptimizer(undefined);
    setLossChartData([]);
  }, []);

  const handleContentLoad = useCallback(({ content, name }: SelectedFile) => {
    setFileContent(content);
    setFileName(name);
    // Reset tokenizer when content changes
    setTokenizer(undefined);
    setModel(undefined);
    setOptimizer(undefined);
    setLossChartData([]);
    setCurrentStep('input');
  }, []);

  const handleProceedToTokenizer = useCallback(() => {
    if (fileContent) {
      setCurrentStep('tokenizer');
    }
  }, [fileContent]);

  const handleModelComplete = useCallback((newModel: LanguageModel) => {
    setModel(newModel);
    setCurrentStep('optimizer');
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

  const bufferRef = useRef<number[]>([]);
  const rafRef = useRef<number | null>(null);
  const workerRef = useRef<Worker>(null);

  useEffect(() => {
    const worker = new TrainWorker();

    worker.onmessage = (event) => {
      const { type, value, message } = event.data;

      if (type === 'loss') {
        bufferRef.current.push(value);

        if (rafRef.current == null) {
          rafRef.current = requestAnimationFrame(() => {
            rafRef.current = null;
            setLossChartData([...bufferRef.current]);
          });
        }
      } else if (type === 'status') {
        console.log('Worker status:', message);
        if (message.includes('completed')) {
          setTrainingInProgress(false);
        }
      } else if (type === 'error') {
        console.error('Worker error:', message);
        setTrainingInProgress(false);
        alert(`Training error: ${message}`);
      } else if (type === 'demo') {
        // Handle original demo messages
        bufferRef.current.push(value);
        if (rafRef.current == null) {
          rafRef.current = requestAnimationFrame(() => {
            rafRef.current = null;
            setLossChartData([...bufferRef.current]);
          });
        }
      }
    };
    workerRef.current = worker;
    return () => {
      worker.terminate();

      if (rafRef.current != null) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, []);

  const deferredPoints = useDeferredValue(lossChartData);

  const randomOutputLossValue = useMemo(() => {
    return tokenizer ? randomOutputLoss(tokenizer.getVocabSize()) : 0;
  }, [tokenizer]);

  const trainModel = useCallback(async () => {
    if (!tokenizer || !model || !optimizer) return;

    setTrainingInProgress(true);

    try {
      // Prepare training data
      const data = tokenizer.encode(fileContent);
      const splitIndex = Math.floor(0.9 * data.length);
      const trainData = data.slice(0, splitIndex);

      console.log(`Training ${model.constructor.name} for ${iterations} iterations...`);

      const iterationTimes: number[] = [];

      // Train in main thread with user's actual configuration
      for (let i = 0; i < iterations; i++) {
        const iterationStart = performance.now();

        const { contexts, outputs } = getBatch(trainData);
        const loss = await optimizer.train(contexts, outputs);

        const iterationEnd = performance.now();
        const iterationTime = iterationEnd - iterationStart;
        iterationTimes.push(iterationTime);

        setLossChartData((prev) => [...prev, loss]);

        console.log(`Iteration ${i + 1}/${iterations}, Loss: ${loss.toFixed(4)} (${iterationTime.toFixed(1)}ms)`);

        // Yield to browser every 10 iterations
        if (i % 10 === 0 && i > 0) {
          await new Promise((resolve) => setTimeout(resolve, 1));
        }
      }

      // Calculate and display average iteration time
      const avgTime = iterationTimes.reduce((sum, time) => sum + time, 0) / iterationTimes.length;
      setAvgIterationTime(avgTime);
      console.log(`Average iteration time: ${avgTime.toFixed(1)}ms`);

      console.log('Training completed!');
    } catch (error) {
      console.error('Training failed:', error);
      alert(`Training failed: ${error.message}`);
    } finally {
      setTrainingInProgress(false);
    }
  }, [tokenizer, model, iterations, optimizer, randomOutputLossValue]);

  const generate = useCallback(async () => {
    if (!tokenizer || !model) return;

    try {
      const initialTokens = tokenizer.encode(initialString);
      const output = await model.generate([initialTokens], 100);
      setGenerateOutput(tokenizer.decode(output[0]));
    } catch (error) {
      console.error('Generation failed:', error);
      setGenerateOutput('Error: Generation failed');
    }
  }, [tokenizer, model, initialString]);

  const renderInputStep = () => (
    <div className="space-y-4">
      <InputConfig onSelectedFileChange={handleContentLoad} selectedFile={{ content: fileContent, name: fileName }} />
      {fileContent && (
        <div className="mt-4">
          <Button onClick={handleProceedToTokenizer}>Next: Create Tokenizer</Button>
        </div>
      )}
    </div>
  );

  const renderTokenizerStep = () => (
    <TokenizerSetup
      fileContent={fileContent}
      fileName={fileName}
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

  const renderTrainStep = () => (
    <div className="space-y-4">
      <div className="mb-6">
        <h2 className="text-xl font-bold">Train & Generate</h2>
        <p className="text-sm text-muted-foreground">Train your model and generate text with it</p>
      </div>

      {tokenizer && model && optimizer && (
        <>
          <div
            className="
              grid grid-cols-1 gap-4
              md:grid-cols-2
            "
          >
            <Label>
              Iterations
              <Input
                className="mt-1"
                onChange={(e) => {
                  setIterations(parseInt(e.target.value));
                }}
                type={'number'}
                value={iterations}
              />
            </Label>
            <Label>
              Initial Context for Generation
              <Input
                className="mt-1"
                onChange={(e) => {
                  setInitialString(e.target.value);
                }}
                placeholder="Enter some text to start generation..."
                value={initialString}
              />
            </Label>
          </div>

          <div className="flex items-center gap-4">
            <Button disabled={trainingInProgress} onClick={() => void trainModel()}>
              {trainingInProgress && <Loader className="mr-2 animate-spin" />}
              {trainingInProgress ? 'Training...' : 'Train Model'}
            </Button>

            <Button disabled={trainingInProgress || !model} onClick={() => void generate()} variant="outline">
              Generate Text
            </Button>

            {avgIterationTime && (
              <span className="text-sm text-muted-foreground">Avg: {avgIterationTime.toFixed(1)}ms/iter</span>
            )}
          </div>

          {generateOutput && (
            <div className="rounded-lg border p-4">
              <div className="mb-2 text-sm font-medium">Generated Output:</div>
              <code
                className="
                  block rounded-sm bg-muted p-2 text-xs whitespace-pre-wrap
                "
              >
                {generateOutput}
              </code>
            </div>
          )}

          <div className="mb-4 flex justify-between">
            <Button onClick={handleBackToOptimizer} variant="outline">
              <ChevronLeft className="mr-1 size-4" />
              Back: Change Optimizer
            </Button>
          </div>

          <ChartContainer
            config={{
              loss: {
                label: 'Loss',
                color: 'var(--chart-1)',
              },
              randomOutput: {
                label: 'Random output loss',
                color: 'var(--chart-2)',
              },
            }}
          >
            <LineChart
              accessibilityLayer
              data={deferredPoints.map((loss, iteration) => ({
                iteration,
                loss,
                randomOutput: randomOutputLossValue,
              }))}
            >
              <CartesianGrid vertical={false} />
              <XAxis axisLine={false} dataKey="iteration" max={iterations} min={0} tickLine={false} tickMargin={8} />
              <YAxis max={randomOutputLossValue * 1.2} min={0} />
              <ChartTooltip content={<ChartTooltipContent hideLabel />} cursor={false} />
              <Line dataKey="loss" dot={false} stroke="var(--chart-1)" strokeWidth={1} type="linear" />
              <Line dataKey="randomOutput" dot={false} stroke="var(--chart-2)" strokeWidth={1} type="linear" />
            </LineChart>
          </ChartContainer>
        </>
      )}
    </div>
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
