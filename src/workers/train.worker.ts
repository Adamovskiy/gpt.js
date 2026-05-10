import { errorMessage } from '@/lib/utils.ts';
import { getBatch } from '@/llm/sampling.ts';

export interface TrainWorkerMessage {
  type: 'train';
  iterations: number;
  trainData: number[];
  modelData: {
    serializedData: unknown;
    type: 'GPTModel';
  };
  optimizerData: {
    serializedData: unknown;
    type: 'UniversalAdamWOptimizer';
  };
  tokenizerData: {
    serializedData: unknown;
    type: string;
  };
}

export type TrainWorkerResponse =
  | {
      message: string;
      type: 'status';
    }
  | {
      message: string;
      type: 'error';
    }
  | {
      iteration: number;
      type: 'loss';
      value: number;
    }
  | {
      avgIterationTime: number;
      modelData: {
        serializedData: unknown;
        type: 'GPTModel';
      };
      optimizerData: {
        serializedData: unknown;
        type: 'UniversalAdamWOptimizer';
      };
      type: 'completed';
    };

self.onmessage = async (event: MessageEvent<TrainWorkerMessage>) => {
  const { iterations, trainData, modelData, optimizerData } = event.data;

  try {
    self.postMessage({
      type: 'status',
      message: `Initializing model and optimizer...`,
    });

    // Dynamically import classes
    const { GPTModel } = await import('../llm/models/GPTModel.ts');
    const { UniversalAdamWOptimizer } = await import('../llm/optimizers/UniversalAdamWOptimizer.ts');

    // Restore model from serialized data
    const model = GPTModel.fromSerializedData(modelData.serializedData as never);

    // Restore optimizer from serialized data
    const optimizer = UniversalAdamWOptimizer.fromSerializedData(optimizerData.serializedData as never, model);

    self.postMessage({
      type: 'status',
      message: `Starting training for ${iterations} iterations...`,
    });

    // Real training loop with timing
    const iterationTimes: number[] = [];
    for (let i = 0; i < iterations; i++) {
      const iterationStart = performance.now();
      const { contexts, outputs } = getBatch(trainData);

      try {
        // Train using the actual optimizer and model
        const loss = await optimizer.train(contexts, outputs);

        const iterationEnd = performance.now();
        const iterationTime = iterationEnd - iterationStart;
        iterationTimes.push(iterationTime);

        self.postMessage({
          type: 'loss',
          value: loss,
          iteration: i,
        });

        // Progress update every 10 iterations
        if (i % 10 === 0) {
          self.postMessage({
            type: 'status',
            message: `Training progress: ${i}/${iterations} iterations completed, Loss: ${loss.toFixed(4)}, iteration time: ${iterationTime.toFixed(4)}`,
          });
        }

        // Small delay to prevent completely blocking the worker
        if (i % 5 === 0) {
          await new Promise((resolve) => setTimeout(resolve, 1));
        }
      } catch (error) {
        console.error('Training step failed:', error);
        self.postMessage({
          type: 'error',
          message: `Training failed at iteration ${i}: ${errorMessage(error)}`,
        });
        break;
      }
    }

    // Calculate average iteration time
    const avgTime = iterationTimes.reduce((sum, time) => sum + time, 0) / iterationTimes.length;

    // Send back the trained model and optimizer
    self.postMessage({
      type: 'completed',
      modelData: {
        type: 'GPTModel',
        serializedData: model.getSerializedData(),
      },
      optimizerData: {
        type: 'UniversalAdamWOptimizer',
        serializedData: optimizer.getSerializedData(),
      },
      avgIterationTime: avgTime,
    });

    self.postMessage({
      type: 'status',
      message: 'Training completed successfully',
    });
  } catch (error) {
    console.error('Training worker failed:', error);
    self.postMessage({
      type: 'error',
      message: errorMessage(error),
    });
  }
};
