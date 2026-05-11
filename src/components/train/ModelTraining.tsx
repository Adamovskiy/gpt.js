import { Loader } from 'lucide-react';
import {
  type Dispatch,
  type SetStateAction,
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from 'recharts';

import type { SelectedFile } from '@/components/input/InputConfig.tsx';
import type { LanguageModel, Optimizer, Tokenizer } from '@/llm/types.ts';
import type { TrainWorkerMessage, TrainWorkerResponse } from '@/workers/train.worker.ts';

import { Button } from '@/components/ui/button.tsx';
import { ChartContainer, ChartTooltip, ChartTooltipContent } from '@/components/ui/chart.tsx';
import { Input } from '@/components/ui/input.tsx';
import { Label } from '@/components/ui/label.tsx';
import { Switch } from '@/components/ui/switch.tsx';
import { errorMessage } from '@/lib/utils.ts';
import { randomOutputLoss } from '@/llm/models/utils.ts';

export function ModelTraining({
  tokenizer,
  model,
  selectedFile,
  optimizer,
  lossChartData,
  setLossChartData,
  setModel,
  setOptimizer,
}: {
  lossChartData: number[];
  model: LanguageModel;
  optimizer: Optimizer;
  selectedFile: SelectedFile;
  setLossChartData: Dispatch<SetStateAction<number[]>>;
  setModel: Dispatch<SetStateAction<LanguageModel | null>>;
  setOptimizer: Dispatch<SetStateAction<Optimizer | null | undefined>>;
  tokenizer: Tokenizer;
}) {
  const [iterations, setIterations] = useState(100);
  const [avgIterationTime, setAvgIterationTime] = useState<number | null>(null);
  const [trainingInProgress, setTrainingInProgress] = useState(false);
  const [statusString, setStatusString] = useState('');
  const [updateChart, setUpdateChart] = useState(true);

  const bufferRef = useRef<number[]>([]);
  const rafRef = useRef<number | null>(null);
  const workerRef = useRef<Worker | null>(null);

  const randomOutputLossValue = useMemo(() => {
    return randomOutputLoss(tokenizer.getVocabSize());
  }, [tokenizer]);

  const trainModel = useCallback(() => {
    if (!workerRef.current) {
      return;
    }

    setTrainingInProgress(true);
    setAvgIterationTime(null);

    // Don't clear chart data - let it accumulate
    bufferRef.current = [];

    try {
      // Prepare training data
      const data = tokenizer.encode(selectedFile.content);
      const trainData = Array.from(data);

      console.log(`Training ${model.constructor.name} for ${iterations} iterations...`);

      // Check if model and optimizer have serialization methods
      if (!('getSerializedData' in model) || !('getSerializedData' in optimizer)) {
        throw new Error('Model or optimizer does not support serialization for worker training');
      }

      // Send training task to worker with serialized data
      const message: TrainWorkerMessage = {
        type: 'train',
        iterations,
        trainData,
        modelData: {
          type: model.constructor.name as
            | 'GPTModel'
            | 'BigramLanguageModel'
            | 'BigramLanguageModelWithFF'
            | 'BigramLanguageModelSingleHeadAttention'
            | 'BigramLanguageModelMultiHeadAttention'
            | 'GPTModelGPU',
          serializedData: model.getSerializedData(),
        },
        optimizerData: {
          type: optimizer.constructor.name as 'UniversalAdamWOptimizer' | 'SDGOptimizer',
          serializedData: optimizer.getSerializedData(),
        },
        tokenizerData: {
          type: tokenizer.constructor.name,
          serializedData: tokenizer.getSerializedData(),
        },
      };

      workerRef.current.postMessage(message);
    } catch (error) {
      console.error('Training failed:', error);
      alert(`Training failed: ${errorMessage(error)}`);
      setTrainingInProgress(false);
    }
  }, [selectedFile, tokenizer, model, optimizer, iterations]);

  useEffect(() => {
    const worker = new Worker(new URL('@/workers/train.worker.ts', import.meta.url), { type: 'module' });

    worker.onmessage = (event: MessageEvent<TrainWorkerResponse>) => {
      const response = event.data;

      if (response.type === 'loss') {
        if (updateChart) {
          bufferRef.current.push(response.value);
        }

        rafRef.current ??= requestAnimationFrame(() => {
          rafRef.current = null;
          setLossChartData((prev) => [...prev, ...bufferRef.current]);
          bufferRef.current = [];
        });
      } else if (response.type === 'status') {
        console.log('Worker status:', response.message);
        setStatusString(response.message);
      } else if (response.type === 'error') {
        console.error('Worker error:', response.message);
        setTrainingInProgress(false);
        alert(`Training error: ${response.message}`);
      } else {
        // if (response.type === 'completed')
        console.log('Training completed, updating model and optimizer with trained versions');

        // Dynamically import and restore model based on type
        const restoreModel = async () => {
          switch (response.modelData.type) {
            case 'BigramLanguageModel': {
              const { BigramLanguageModel } = await import('@/llm/models/BigramLanguageModel.ts');
              return BigramLanguageModel.fromSerializedData(response.modelData.serializedData);
            }
            case 'BigramLanguageModelMultiHeadAttention': {
              const { BigramLanguageModelMultiHeadAttention } =
                await import('@/llm/models/BigramLanguageModelMultiHeadAttention.ts');
              return BigramLanguageModelMultiHeadAttention.fromSerializedData(response.modelData.serializedData);
            }
            case 'BigramLanguageModelSingleHeadAttention': {
              const { BigramLanguageModelSingleHeadAttention } =
                await import('@/llm/models/BigramLanguageModelSingleHeadAttention.ts');
              return BigramLanguageModelSingleHeadAttention.fromSerializedData(response.modelData.serializedData);
            }
            case 'BigramLanguageModelWithFF': {
              const { BigramLanguageModelWithFF } = await import('@/llm/models/BigramLanguageModelWithFF.ts');
              return BigramLanguageModelWithFF.fromSerializedData(response.modelData.serializedData);
            }
            case 'GPTModel': {
              const { GPTModel } = await import('@/llm/models/GPTModel.ts');
              return GPTModel.fromSerializedData(response.modelData.serializedData as never);
            }
            case 'GPTModelGPU': {
              const { GPTModelGPU } = await import('@/llm/models/gpu/GPTModelGPU.ts');
              return GPTModelGPU.fromSerializedData(response.modelData.serializedData);
            }
            default: {
              const exhaustiveCheck: never = response.modelData.type;
              throw new Error(`Unsupported model type: ${exhaustiveCheck}`);
            }
          }
        };

        // Dynamically import and restore optimizer based on type
        const restoreOptimizer = async (trainedModel: LanguageModel) => {
          switch (response.optimizerData.type) {
            case 'SDGOptimizer': {
              const { SDGOptimizer } = await import('@/llm/optimizers/SDGOptimizer.ts');
              return SDGOptimizer.fromSerializedData(response.optimizerData.serializedData as never, trainedModel);
            }
            case 'UniversalAdamWOptimizer': {
              const { UniversalAdamWOptimizer } = await import('@/llm/optimizers/UniversalAdamWOptimizer.ts');
              return UniversalAdamWOptimizer.fromSerializedData(
                response.optimizerData.serializedData as never,
                trainedModel,
              );
            }
            default: {
              const exhaustiveCheck: never = response.optimizerData.type;
              throw new Error(`Unsupported optimizer type: ${exhaustiveCheck}`);
            }
          }
        };

        // Restore both model and optimizer
        void restoreModel().then(async (trainedModel) => {
          const trainedOptimizer = await restoreOptimizer(trainedModel);

          setModel(trainedModel);
          setOptimizer(trainedOptimizer);
          setAvgIterationTime(response.avgIterationTime);
          setTrainingInProgress(false);
        });
      }
    };

    workerRef.current = worker;

    return () => {
      worker.terminate();

      if (rafRef.current != null) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, [setLossChartData, setModel, setOptimizer, updateChart]);

  const deferredPoints = useDeferredValue(lossChartData);

  return (
    <>
      <div className="space-y-4">
        <Label>
          Training Iterations
          <Input
            className="mt-1"
            onChange={(e) => {
              setIterations(parseInt(e.target.value));
            }}
            type={'number'}
            value={iterations}
          />
        </Label>

        <div className="flex items-center gap-4">
          <Button
            disabled={trainingInProgress}
            onClick={() => {
              trainModel();
            }}
          >
            {trainingInProgress && <Loader className="mr-2 animate-spin" />}
            {trainingInProgress ? 'Training...' : 'Train Model'}
          </Button>

          <div className="flex items-center space-x-2" title="Turn it off to save some RAM">
            <Switch
              checked={updateChart}
              disabled={trainingInProgress}
              id="update-chart"
              onCheckedChange={setUpdateChart}
            />
            <Label htmlFor="update-chart">With chart update</Label>
          </div>

          {avgIterationTime && (
            <span className="text-sm text-muted-foreground">Avg: {avgIterationTime.toFixed(1)}ms/iter</span>
          )}
        </div>
        <div>{statusString}</div>
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
  );
}
