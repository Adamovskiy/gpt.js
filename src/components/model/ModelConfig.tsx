import type { LanguageModel } from '../../types.ts';
import {
  GPTModel,
  GPTModelGPU,
  BigramLanguageModel,
  BigramLanguageModelSingleHeadAttention,
  BigramLanguageModelMultiHeadAttention,
} from '../../tfOps.ts';
import { blockSize } from '../../sampling.ts';
import { Button } from '../ui/button.tsx';
import { Badge } from '../ui/badge.tsx';
import { useState } from 'react';

export function ModelConfig({
  vocabSize,
  model,
  setModel,
  onModelChange,
}: {
  vocabSize: number;
  model: LanguageModel | undefined;
  setModel: (model: LanguageModel) => void;
  onModelChange?: () => void;
}) {
  const [modelType, setModelType] = useState<string>('gpt-cpu');
  const [isInitializing, setIsInitializing] = useState(false);

  const createModel = async (type: string) => {
    setIsInitializing(true);
    try {
      const numberEmbeddingDimensions = 32;
      const numHeads = 2;
      const numLayers = 2;

      let modelInstance: LanguageModel;

      switch (type) {
        case 'bigram':
          modelInstance = new BigramLanguageModel(vocabSize, numberEmbeddingDimensions, blockSize);
          break;
        case 'single-head':
          modelInstance = new BigramLanguageModelSingleHeadAttention(vocabSize, numberEmbeddingDimensions, blockSize);
          break;
        case 'multi-head':
          modelInstance = new BigramLanguageModelMultiHeadAttention(
            vocabSize,
            numberEmbeddingDimensions,
            blockSize,
            numHeads,
          );
          break;
        case 'gpt-cpu':
          modelInstance = new GPTModel(vocabSize, numberEmbeddingDimensions, blockSize, numHeads, numLayers);
          break;
        case 'gpt-gpu':
          const gpuModel = new GPTModelGPU(vocabSize, numberEmbeddingDimensions, blockSize, numHeads, numLayers);
          await gpuModel.initializeGPU();
          modelInstance = gpuModel;
          break;
        default:
          throw new Error(`Unknown model type: ${type}`);
      }

      setModel(modelInstance);
      onModelChange?.(); // Notify parent about model change
    } catch (error) {
      console.error('Failed to create model:', error);
      alert(`Failed to create ${type} model: ${error.message}`);
    } finally {
      setIsInitializing(false);
    }
  };

  const modelTypes = [
    { value: 'bigram', label: 'Bigram Language Model', description: 'Simple bigram model with embeddings' },
    { value: 'single-head', label: 'Single Head Attention', description: 'Bigram + single attention head' },
    { value: 'multi-head', label: 'Multi Head Attention', description: 'Bigram + multi-head attention' },
    { value: 'gpt-cpu', label: 'GPT (CPU)', description: 'Full transformer with multiple layers (CPU)' },
    { value: 'gpt-gpu', label: 'GPT (GPU)', description: 'Full transformer with GPU acceleration (WebGPU)' },
  ];

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Model Configuration</h3>

      <div className="space-y-4">
        {/* Model Type Selection */}
        <div>
          <label className="text-sm font-medium">Model Type</label>
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            className="w-full mt-1 p-2 border rounded-md"
          >
            {modelTypes.map((type) => (
              <option key={type.value} value={type.value}>
                {type.label} - {type.description}
              </option>
            ))}
          </select>
        </div>

        {/* Current Model Info */}
        {model && (
          <div className="border p-4 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-sm font-medium">{(model as any).constructor.name}</span>
              <Badge variant="outline">{model.getParameters().length} parameters</Badge>
              {(model as any).isGPU && <Badge variant="secondary">GPU</Badge>}
            </div>
            <div className="text-xs text-muted-foreground">
              {
                modelTypes.find(
                  (t) =>
                    (t.value === 'bigram' && model.constructor.name === 'BigramLanguageModel') ||
                    (t.value === 'single-head' &&
                      model.constructor.name === 'BigramLanguageModelSingleHeadAttention') ||
                    (t.value === 'multi-head' && model.constructor.name === 'BigramLanguageModelMultiHeadAttention') ||
                    (t.value === 'gpt-cpu' && model.constructor.name === 'GPTModel') ||
                    (t.value === 'gpt-gpu' && model.constructor.name === 'GPTModelGPU'),
                )?.description
              }
            </div>
          </div>
        )}

        {/* Create Model Button */}
        <Button onClick={() => createModel(modelType)} disabled={isInitializing} className="w-full">
          {isInitializing ? 'Creating...' : model ? 'Re-create' : 'Create'} Model
        </Button>

        {/* WebGPU Notice */}
        {modelType === 'gpt-gpu' && (
          <p className="text-sm text-muted-foreground">GPU model requires WebGPU support (Chrome 113+, Edge 113+)</p>
        )}
      </div>
    </div>
  );
}
