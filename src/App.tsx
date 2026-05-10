import { useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';
import { seed } from './lib/random.ts';
import { getBatch } from './llm/sampling.ts';
import { type Optimizer } from './llm/optimizers/utils.ts';
import { Button } from '@/components/ui/button.tsx';
import { Loader, ChevronLeft } from 'lucide-react';
import { ChartContainer, ChartTooltip, ChartTooltipContent } from './components/ui/chart.tsx';
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from 'recharts';
import { TokenizerSetup } from '@/components/tokenizer/TokenizerSetup.tsx';
import { ModelConfig } from './components/model/ModelConfig.tsx';
import type { LanguageModel, Tokenizer } from './llm/types.ts';
import { Label } from './components/ui/label.tsx';
import { Input } from './components/ui/input.tsx';
import { OptimizerConfig } from './components/optimizer/OptimizerConfig.tsx';
import TrainWorker from './workers/train.worker.ts?worker';
import { InputConfig, type SelectedFile } from './components/input/InputConfig.tsx';
import { randomOutputLoss } from './llm/models/utils.ts';

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
      <InputConfig selectedFile={{ content: fileContent, name: fileName }} onSelectedFileChange={handleContentLoad} />
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
      onComplete={handleTokenizerComplete}
      onBack={handleBackToInput}
    />
  );

  const renderModelStep = () => (
    <div>
      {tokenizer && (
        <ModelConfig
          vocabSize={tokenizer.getVocabSize()}
          onComplete={handleModelComplete}
          onBack={() => setCurrentStep('tokenizer')}
        />
      )}
    </div>
  );

  const renderOptimizerStep = () => (
    <div>
      {model && <OptimizerConfig model={model} onComplete={handleOptimizerComplete} onBack={handleBackToModel} />}
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
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Label>
              Iterations
              <Input
                value={iterations}
                onChange={(e) => setIterations(parseInt(e.target.value))}
                type={'number'}
                className="mt-1"
              />
            </Label>
            <Label>
              Initial Context for Generation
              <Input
                value={initialString}
                onChange={(e) => setInitialString(e.target.value)}
                placeholder="Enter some text to start generation..."
                className="mt-1"
              />
            </Label>
          </div>

          <div className="flex items-center gap-4">
            <Button onClick={() => trainModel()} disabled={trainingInProgress}>
              {trainingInProgress && <Loader className="animate-spin mr-2" />}
              {trainingInProgress ? 'Training...' : 'Train Model'}
            </Button>

            <Button onClick={() => generate()} disabled={trainingInProgress || !model} variant="outline">
              Generate Text
            </Button>

            {avgIterationTime && (
              <span className="text-sm text-muted-foreground">Avg: {avgIterationTime.toFixed(1)}ms/iter</span>
            )}
          </div>

          {generateOutput && (
            <div className="border p-4 rounded-lg">
              <div className="text-sm font-medium mb-2">Generated Output:</div>
              <code className="text-xs bg-muted p-2 rounded block whitespace-pre-wrap">{generateOutput}</code>
            </div>
          )}

          <div className="flex justify-between mb-4">
            <Button variant="outline" onClick={handleBackToOptimizer}>
              <ChevronLeft className="w-4 h-4 mr-1" />
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
                iteration: iteration,
                loss: loss,
                randomOutput: randomOutputLossValue,
              }))}
            >
              <CartesianGrid vertical={false} />
              <XAxis dataKey="iteration" tickLine={false} axisLine={false} tickMargin={8} max={iterations} min={0} />
              <YAxis min={0} max={randomOutputLossValue * 1.2} />
              <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
              <Line dataKey="loss" type="linear" stroke="var(--chart-1)" strokeWidth={1} dot={false} />
              <Line dataKey="randomOutput" type="linear" stroke="var(--chart-2)" strokeWidth={1} dot={false} />
            </LineChart>
          </ChartContainer>
        </>
      )}
    </div>
  );

  return (
    <main className="max-w-3xl mx-auto px-2 py-4 flex flex-col gap-4">
      {currentStep === 'input' && renderInputStep()}
      {currentStep === 'tokenizer' && renderTokenizerStep()}
      {currentStep === 'model' && renderModelStep()}
      {currentStep === 'optimizer' && renderOptimizerStep()}
      {currentStep === 'train' && renderTrainStep()}
    </main>
  );
}

export default App;
