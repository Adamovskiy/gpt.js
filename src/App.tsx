import { useCallback, useMemo, useState } from 'react';
import fileContent from './voynaimir.txt?raw';
import { seed } from './random.ts';
import { CharTokenizer } from './tokenizer.ts';
import { randomOutputLoss } from './tfOps.ts';
import { getBatch } from './sampling.ts';
import { type Optimizer } from './optimizers.ts';
import { Button } from '@/components/ui/button.tsx';
import { Loader } from 'lucide-react';
import { ChartContainer, ChartTooltip, ChartTooltipContent } from './components/ui/chart.tsx';
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from 'recharts';
import { TokenizerDemo } from '@/components/tokenizer/TokenizerDemo.tsx';
import { Vocabulary } from '@/components/tokenizer/Vocabulary.tsx';
import { ModelConfig } from './components/model/ModelConfig.tsx';
import type { LanguageModel } from './types.ts';
import { Label } from './components/ui/label.tsx';
import { Input } from './components/ui/input.tsx';
import { OptimizerConfig } from './components/optimizer/OptimizerConfig.tsx';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card.tsx';

function App() {
  const [lossChartData, setLossChartData] = useState<number[]>([]);
  const [tokenizer, setTokenizer] = useState<CharTokenizer | undefined>();
  const [model, setModel] = useState<LanguageModel | undefined>();
  const [optimizer, setOptimizer] = useState<Optimizer | undefined>();
  const [trainingInProgress, setTrainingInProgress] = useState(false);
  const [iterations, setIterations] = useState(100);
  const [initialString, setInitialString] = useState('');
  const [generateOutput, setGenerateOutput] = useState<string>();

  const initTokenizer = useCallback(() => {
    if (tokenizer) return;
    seed(42);
    setTokenizer(new CharTokenizer(fileContent));
  }, []);

  const trainModel = useCallback(() => {
    if (!tokenizer || !model || !optimizer) return;
    setTrainingInProgress(true);
    setTimeout(() => {
      const data = tokenizer.encode(fileContent);
      const splitIndex = 0.9 * data.length;
      const trainData = data.slice(0, splitIndex);

      let loss: number;
      for (let i = 0; i < iterations; i++) {
        const { contexts, outputs } = getBatch(trainData);
        loss = optimizer.train(contexts, outputs);
        const lossCapture = loss;
        setLossChartData((data) => [...data, lossCapture]);
        console.log(`Loss: ${loss} (perfect - 0, random - ${randomOutputLoss(tokenizer.getVocabSize())})`);
      }
    }, 0);
    setTrainingInProgress(false);
  }, [tokenizer, model, iterations, optimizer]);

  const generate = useCallback(() => {
    if (!tokenizer || !model) return;

    const initialTokens = tokenizer.encode(initialString);
    const output = model.generate([initialTokens], 100);
    setGenerateOutput(tokenizer.decode(output[0]));
  }, [tokenizer, model, initialString]);

  const randomOutputLossValue = useMemo(() => {
    return tokenizer ? randomOutputLoss(tokenizer.getVocabSize()) : 0;
  }, [tokenizer]);

  return (
    <main className="max-w-3xl mx-auto px-2 py-4 flex flex-col gap-4">
      <div>
        <Card>
          <CardHeader>
            <CardTitle>Input preview: voynaimir.txt</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="bg-muted p-2 whitespace-pre-wrap">{fileContent.slice(0, 200)}...</pre>
          </CardContent>
        </Card>
        {!tokenizer && (
          <div>
            <Button onClick={() => initTokenizer()}>Init model</Button>
          </div>
        )}
      </div>
      {tokenizer && (
        <>
          <Vocabulary tokenizer={tokenizer} />
          <TokenizerDemo tokenizer={tokenizer} />
          <ModelConfig vocabSize={tokenizer.getVocabSize()} model={model} setModel={setModel} />

          {model && (
            <div>
              <OptimizerConfig optimizer={optimizer} setOptimizer={setOptimizer} model={model} />
              {optimizer && (
                <>
                  {' '}
                  <Label>
                    Iterations
                    <Input
                      value={iterations}
                      onChange={(e) => setIterations(parseInt(e.target.value))}
                      type={'number'}
                    />
                  </Label>
                  <Button onClick={() => trainModel()} disabled={trainingInProgress}>
                    {trainingInProgress && <Loader className="animate-spin" />}Train model
                  </Button>
                </>
              )}
              <Label>
                Initial context
                <Input value={initialString} onChange={(e) => setInitialString(e.target.value)} />
              </Label>
              <Button onClick={() => generate()} disabled={trainingInProgress}>
                Generate
              </Button>
              {generateOutput && (
                <div>
                  <div>Output:</div>
                  <code>{generateOutput}</code>
                </div>
              )}
            </div>
          )}
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
              data={lossChartData.map((loss, iteration) => ({
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
    </main>
  );
}

export default App;
