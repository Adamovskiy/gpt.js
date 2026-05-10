import type { LanguageModel } from '../../llm/types.ts';
import { blockSize } from '../../llm/sampling.ts';
import { Button } from '../ui/button.tsx';
import { Badge } from '../ui/badge.tsx';
import { Card } from '../ui/card.tsx';
import { Input } from '../ui/input.tsx';
import { Label } from '../ui/label.tsx';
import { ChevronLeft, ChevronRight, Loader } from 'lucide-react';
import { useState, useCallback } from 'react';
import { BigramLanguageModel } from '../../llm/models/BigramLanguageModel.ts';
import { BigramLanguageModelMultiHeadAttention } from '../../llm/models/BigramLanguageModelMultiHeadAttention.ts';
import { BigramLanguageModelSingleHeadAttention } from '../../llm/models/BigramLanguageModelSingleHeadAttention.ts';
import { GPTModelGPU } from '../../llm/models/gpu/GPTModelGPU.ts';
import { GPTModel } from '../../llm/models/GPTModel.ts';

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
        case 'single-head':
          modelInstance = new BigramLanguageModelSingleHeadAttention(vocabSize, embeddingDim, blockSize);
          break;
        case 'multi-head':
          modelInstance = new BigramLanguageModelMultiHeadAttention(vocabSize, embeddingDim, blockSize, numHeads);
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
                type="number"
                min="1"
                max="512"
                value={embeddingDim}
                onChange={(e) => setEmbeddingDim(parseInt(e.target.value) || 32)}
                disabled={isCreating}
                className="mt-1"
              />
            </Label>
            <div className="text-sm text-muted-foreground">
              Number of embedding dimensions. Higher values can capture more complex patterns but require more memory.
            </div>
          </div>
        );

      case 'multi-head': {
        return (
          <div className="space-y-3">
            <Label>
              Embedding Dimensions
              <Input
                type="number"
                min="1"
                max="512"
                value={embeddingDim}
                onChange={(e) => setEmbeddingDim(parseInt(e.target.value) || 32)}
                disabled={isCreating}
                className="mt-1"
              />
            </Label>
            <Label>
              Number of Attention Heads
              <Input
                type="number"
                min="1"
                max="16"
                value={numHeads}
                onChange={(e) => setNumHeads(parseInt(e.target.value) || 2)}
                disabled={isCreating}
                className="mt-1"
              />
            </Label>
            <div className="text-sm text-muted-foreground">
              Multiple attention heads allow the model to attend to different aspects of the sequence simultaneously.
            </div>
          </div>
        );
      }

      case 'gpt-cpu':
      case 'gpt-gpu':
        return (
          <div className="space-y-3">
            <Label>
              Embedding Dimensions
              <Input
                type="number"
                min="1"
                max="512"
                value={embeddingDim}
                onChange={(e) => setEmbeddingDim(parseInt(e.target.value) || 32)}
                disabled={isCreating}
                className="mt-1"
              />
            </Label>
            <Label>
              Number of Attention Heads
              <Input
                type="number"
                min="1"
                max="16"
                value={numHeads}
                onChange={(e) => setNumHeads(parseInt(e.target.value) || 2)}
                disabled={isCreating}
                className="mt-1"
              />
            </Label>
            <Label>
              Number of Transformer Layers
              <Input
                type="number"
                min="1"
                max="12"
                value={numLayers}
                onChange={(e) => setNumLayers(parseInt(e.target.value) || 2)}
                disabled={isCreating}
                className="mt-1"
              />
            </Label>
            <div className="text-sm text-muted-foreground">
              Full transformer architecture with multiple layers. More layers can learn more complex patterns but are
              slower to train.
            </div>
          </div>
        );

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
              id="model-type-select"
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              disabled={isCreating}
              className="w-full p-2 border border-input rounded-md bg-background disabled:opacity-50"
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
            <div className="p-3 text-sm text-yellow-800 bg-yellow-50 rounded-md">
              ⚠️ GPU model requires WebGPU support (Chrome 113+, Edge 113+)
            </div>
          )}
        </div>

        {model && (
          <Card className="p-4 bg-green-50">
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
          <Button variant="outline" onClick={onBack} disabled={isCreating}>
            <ChevronLeft className="w-4 h-4 mr-1" />
            Back
          </Button>

          {!model ? (
            <Button onClick={createModel} disabled={isCreating}>
              {isCreating ? (
                <>
                  <Loader className="w-4 h-4 mr-2 animate-spin" />
                  Creating Model...
                </>
              ) : (
                <>Create Model</>
              )}
            </Button>
          ) : (
            <Button onClick={handleFinish}>
              Next: Create Optimizer
              <ChevronRight className="w-4 h-4 ml-1" />
            </Button>
          )}
        </div>
      </div>
    </Card>
  );
}
