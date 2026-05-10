import { ChevronLeft, ChevronRight, Loader } from 'lucide-react';
import { useCallback, useState } from 'react';

import type { LanguageModel } from '../../llm/types.ts';

import { BigramLanguageModel } from '../../llm/models/BigramLanguageModel.ts';
import { BigramLanguageModelMultiHeadAttention } from '../../llm/models/BigramLanguageModelMultiHeadAttention.ts';
import { BigramLanguageModelSingleHeadAttention } from '../../llm/models/BigramLanguageModelSingleHeadAttention.ts';
import { GPTModel } from '../../llm/models/GPTModel.ts';
import { GPTModelGPU } from '../../llm/models/gpu/GPTModelGPU.ts';
import { blockSize } from '../../llm/sampling.ts';
import { Badge } from '../ui/badge.tsx';
import { Button } from '../ui/button.tsx';
import { Card } from '../ui/card.tsx';
import { Input } from '../ui/input.tsx';
import { Label } from '../ui/label.tsx';

export interface ModelConfigProps {
  vocabSize: number;
  onComplete: (model: LanguageModel) => void;
  onBack: () => void;
}

export function ModelConfig({ vocabSize, onComplete, onBack }: ModelConfigProps) {
  const [modelType, setModelType] = useState<string>('gpt-cpu');
  const [isCreating, setIsCreating] = useState(false);
  const [model, setModel] = useState<LanguageModel | undefined>();

  // Model parameters with defaults
  const [embeddingDim, setEmbeddingDim] = useState(32);
  const [numHeads, setNumHeads] = useState(2);
  const [numLayers, setNumLayers] = useState(2);

  const createModel = useCallback(async () => {
    if (isCreating) return;

    setIsCreating(true);
    try {
      let modelInstance: LanguageModel;

      switch (modelType) {
        case 'bigram':
          modelInstance = new BigramLanguageModel(vocabSize, embeddingDim, blockSize);
          break;
        case 'gpt-cpu':
          modelInstance = new GPTModel(vocabSize, embeddingDim, blockSize, numHeads, numLayers);
          break;
        case 'gpt-gpu': {
          const gpuModel = new GPTModelGPU(vocabSize, embeddingDim, blockSize, numHeads, numLayers);
          await gpuModel.initializeGPU();
          modelInstance = gpuModel;
          break;
        }
        case 'multi-head':
          modelInstance = new BigramLanguageModelMultiHeadAttention(vocabSize, embeddingDim, blockSize, numHeads);
          break;
        case 'single-head':
          modelInstance = new BigramLanguageModelSingleHeadAttention(vocabSize, embeddingDim, blockSize);
          break;
        default:
          throw new Error(`Unknown model type: ${modelType}`);
      }

      setModel(modelInstance);
    } catch (error) {
      console.error('Failed to create model:', error);
      alert(`Failed to create ${modelType} model: ${error.message}`);
    } finally {
      setIsCreating(false);
    }
  }, [vocabSize, embeddingDim, numHeads, numLayers, modelType, isCreating]);

  const handleFinish = useCallback(() => {
    if (model) {
      onComplete(model);
    }
  }, [model, onComplete]);

  const modelTypes = [
    { value: 'bigram', label: 'Bigram Language Model', description: 'Simple bigram model with embeddings' },
    { value: 'single-head', label: 'Single Head Attention', description: 'Bigram + single attention head' },
    { value: 'multi-head', label: 'Multi Head Attention', description: 'Bigram + multi-head attention' },
    { value: 'gpt-cpu', label: 'GPT (CPU)', description: 'Full transformer with multiple layers (CPU)' },
    { value: 'gpt-gpu', label: 'GPT (GPU)', description: 'Full transformer with GPU acceleration (WebGPU)' },
  ];

  const renderModelParameters = () => {
    switch (modelType) {
      case 'bigram':
      case 'single-head':
        return (
          <div className="space-y-3">
            <Label>
              Embedding Dimensions
              <Input
                className="mt-1"
                disabled={isCreating}
                max="512"
                min="1"
                onChange={(e) => {
                  setEmbeddingDim(parseInt(e.target.value) || 32);
                }}
                type="number"
                value={embeddingDim}
              />
            </Label>
            <div className="text-sm text-muted-foreground">
              Number of embedding dimensions. Higher values can capture more complex patterns but require more memory.
            </div>
          </div>
        );

      case 'gpt-cpu':

      case 'gpt-gpu':
        return (
          <div className="space-y-3">
            <Label>
              Embedding Dimensions
              <Input
                className="mt-1"
                disabled={isCreating}
                max="512"
                min="1"
                onChange={(e) => {
                  setEmbeddingDim(parseInt(e.target.value) || 32);
                }}
                type="number"
                value={embeddingDim}
              />
            </Label>
            <Label>
              Number of Attention Heads
              <Input
                className="mt-1"
                disabled={isCreating}
                max="16"
                min="1"
                onChange={(e) => {
                  setNumHeads(parseInt(e.target.value) || 2);
                }}
                type="number"
                value={numHeads}
              />
            </Label>
            <Label>
              Number of Transformer Layers
              <Input
                className="mt-1"
                disabled={isCreating}
                max="12"
                min="1"
                onChange={(e) => {
                  setNumLayers(parseInt(e.target.value) || 2);
                }}
                type="number"
                value={numLayers}
              />
            </Label>
            <div className="text-sm text-muted-foreground">
              Full transformer architecture with multiple layers. More layers can learn more complex patterns but are
              slower to train.
            </div>
          </div>
        );
      case 'multi-head': {
        return (
          <div className="space-y-3">
            <Label>
              Embedding Dimensions
              <Input
                className="mt-1"
                disabled={isCreating}
                max="512"
                min="1"
                onChange={(e) => {
                  setEmbeddingDim(parseInt(e.target.value) || 32);
                }}
                type="number"
                value={embeddingDim}
              />
            </Label>
            <Label>
              Number of Attention Heads
              <Input
                className="mt-1"
                disabled={isCreating}
                max="16"
                min="1"
                onChange={(e) => {
                  setNumHeads(parseInt(e.target.value) || 2);
                }}
                type="number"
                value={numHeads}
              />
            </Label>
            <div className="text-sm text-muted-foreground">
              Multiple attention heads allow the model to attend to different aspects of the sequence simultaneously.
            </div>
          </div>
        );
      }

      default:
        return null;
    }
  };

  return (
    <Card className="p-6">
      <div className="space-y-6">
        <div>
          <h3 className="text-lg font-semibold">Configure Model</h3>
          <p className="text-sm text-muted-foreground">Choose the model architecture and configure its parameters</p>
        </div>

        <div className="space-y-4">
          <div className="space-y-3">
            <Label htmlFor="model-type-select">Model Type</Label>
            <select
              className="
                w-full rounded-md border border-input bg-background p-2
                disabled:opacity-50
              "
              disabled={isCreating}
              id="model-type-select"
              onChange={(e) => {
                setModelType(e.target.value);
              }}
              value={modelType}
            >
              {modelTypes.map((type) => (
                <option key={type.value} value={type.value}>
                  {type.label}
                </option>
              ))}
            </select>

            <div className="text-sm text-muted-foreground">
              {modelTypes.find((t) => t.value === modelType)?.description}
            </div>
          </div>

          {renderModelParameters()}

          {modelType === 'gpt-gpu' && (
            <div className="rounded-md bg-yellow-50 p-3 text-sm text-yellow-800">
              ⚠️ GPU model requires WebGPU support (Chrome 113+, Edge 113+)
            </div>
          )}
        </div>

        {model && (
          <Card className="bg-green-50 p-4">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">
                  {(model as { constructor: { name: string } }).constructor.name}
                </span>
                <Badge variant="outline">{model.getParameters().length} parameters</Badge>
                {(model as { isGPU?: boolean }).isGPU && <Badge variant="secondary">GPU</Badge>}
              </div>
              <div className="text-xs text-muted-foreground">
                Model created successfully with vocabulary size: {vocabSize}
              </div>
            </div>
          </Card>
        )}

        <div className="flex justify-between">
          <Button disabled={isCreating} onClick={onBack} variant="outline">
            <ChevronLeft className="mr-1 size-4" />
            Back
          </Button>

          {!model ? (
            <Button disabled={isCreating} onClick={createModel}>
              {isCreating ? (
                <>
                  <Loader className="mr-2 size-4 animate-spin" />
                  Creating Model...
                </>
              ) : (
                <>Create Model</>
              )}
            </Button>
          ) : (
            <Button onClick={handleFinish}>
              Next: Create Optimizer
              <ChevronRight className="ml-1 size-4" />
            </Button>
          )}
        </div>
      </div>
    </Card>
  );
}
