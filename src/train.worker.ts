// Simple worker for future GPU operations if needed

import { getBatch } from './llm/sampling.ts';

self.onmessage = async (event) => {
  const { type, model, tokenizer, optimizer, iterations, trainData } = event.data;

  if (type === 'train') {
    if (!model || !tokenizer || !optimizer) {
      self.postMessage({
        type: 'error',
        message: 'Missing model, tokenizer, or optimizer',
      });
      return;
    }

    try {
      self.postMessage({
        type: 'status',
        message: `Training ${model.constructor?.name || 'model'} for ${iterations} iterations...`,
      });

      // Use the actual user-configured model, tokenizer, and optimizer
      for (let i = 0; i < iterations; i++) {
        const { contexts, outputs } = getBatch(trainData);

        try {
          // Train using the user's optimizer and model
          const loss = optimizer.train(contexts, outputs);

          self.postMessage({
            type: 'loss',
            value: loss,
            iteration: i,
          });

          // Small delay to show progress
          if (i % 10 === 0) {
            await new Promise((resolve) => setTimeout(resolve, 10));
          }
        } catch (error) {
          console.error('Training step failed:', error);
          self.postMessage({
            type: 'error',
            message: `Training failed at iteration ${i}: ${error.message}`,
          });
          break;
        }
      }

      self.postMessage({
        type: 'status',
        message: 'Training completed',
      });
    } catch (error) {
      console.error('Training failed:', error);
      self.postMessage({
        type: 'error',
        message: `Training failed: ${error.message}`,
      });
    }
  } else if (type === 'demo') {
    // Simple demo functionality
    for (let i = 0; i < 1000; i++) {
      const value = heavyCalculation(i);
      self.postMessage({
        type: 'demo',
        index: i,
        value,
      });
    }
  }
};

function heavyCalculation(i: number) {
  return Math.sin(i);
}
