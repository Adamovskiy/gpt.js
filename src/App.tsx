import { useCallback, useState } from 'react';
import fileContent from './voynaimir.txt?raw';
import { seed } from './random.ts';
import { CharTokenizer } from './tokenizer.ts';
import { GPTModel } from './tfOps.ts';
import { blockSize, getBatch } from './sampling.ts';
import { UniversalAdamWOptimizer } from './optimizers.ts';

function App() {
  const [lossChartData, setLossChartData] = useState<number[]>([]);
  const [tokenizer, setTokenizer] = useState<CharTokenizer>();

  const initTokenizer = useCallback(() => {
    if (tokenizer) return;
    seed(42);
    setTokenizer(new CharTokenizer(fileContent));
  }, []);

  const trainModel = useCallback(() => {
    if (!tokenizer) return;
    const numberEmbeddingDimensions = 32;

    const data = tokenizer.encode(fileContent);
    const splitIndex = 0.9 * data.length;
    const trainData = data.slice(0, splitIndex);

    const numHeads = 2; // Reduce heads
    const numLayers = 2; // Reduce layers
    const model = new GPTModel(tokenizer.getVocabSize(), numberEmbeddingDimensions, blockSize, numHeads, numLayers);

    // Learning loop - much smaller learning rate
    const optimizer = new UniversalAdamWOptimizer(model, 3e-4, 0.9, 0.999, 1e-8, 0.01);
    let loss: number;
    for (let i = 0; i < 1000; i++) {
      const { contexts, outputs } = getBatch(trainData);
      loss = optimizer.train(contexts, outputs);
      setLossChartData((data) => [...data, loss]);
      console.log(`Loss: ${loss} (perfect - 0, random - ${-Math.log(1 / tokenizer.getVocabSize())})`);
    }

    const initialTokens = tokenizer.encode('A');
    const output = model.generate([initialTokens], 100);
    console.log(tokenizer.decode(output[0]));
  }, []);

  return (
    <>
      <h2 className="text-3xl">Input preview:</h2>
      <code>{fileContent.slice(0, 200)}...</code>
      {!tokenizer && (
        <div>
          <button onClick={() => initTokenizer()}>Init model</button>
        </div>
      )}
      {tokenizer && (
        <>
          <h2 className="text-3xl">Tokenizer:</h2>
          <div>
            Vocabulary size: {tokenizer.getVocabSize()}
            <table>
              <tbody>
                <tr>
                  <th>Value</th>
                  <th>Token</th>
                </tr>
                {tokenizer.vocabulary.map((token, idx) => (
                  <tr>
                    <td>{idx}</td>
                    <td>{JSON.stringify(token).substring(1, JSON.stringify(token).length - 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <button onClick={() => trainModel()}>Train model</button>
          <div>{JSON.stringify(lossChartData)}</div>
        </>
      )}
    </>
  );
}

export default App;
