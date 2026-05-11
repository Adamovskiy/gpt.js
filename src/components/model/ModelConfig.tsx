import { useCallback, useEffect, useState } from 'react';

import type { LanguageModel } from '@/llm/types.ts';

import { BackButton } from '@/components/layout/BackButton.tsx';
import { NextButton } from '@/components/layout/NextButton.tsx';
import { StepLayout } from '@/components/layout/StepLayout.tsx';
import { Card } from '@/components/ui/card.tsx';
import { Field, FieldLabel } from '@/components/ui/field.tsx';
import { Input } from '@/components/ui/input.tsx';
import { Label } from '@/components/ui/label.tsx';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { errorMessage } from '@/lib/utils.ts';
import { BigramLanguageModel } from '@/llm/models/BigramLanguageModel.ts';
import { BigramLanguageModelMultiHeadAttention } from '@/llm/models/BigramLanguageModelMultiHeadAttention.ts';
import { BigramLanguageModelSingleHeadAttention } from '@/llm/models/BigramLanguageModelSingleHeadAttention.ts';
import { BigramLanguageModelWithFF } from '@/llm/models/BigramLanguageModelWithFF.ts';
import { GPTModel } from '@/llm/models/GPTModel.ts';
import { GPTModelGPU } from '@/llm/models/gpu/GPTModelGPU.ts';
import { blockSize } from '@/llm/sampling.ts';

interface BigramConfig {
  type: 'bigram';
  embeddingDim: number;
}

interface SingleHeadConfig {
  type: 'single-head';
  embeddingDim: number;
}

interface MultiHeadConfig {
  type: 'multi-head';
  embeddingDim: number;
  numHeads: number;
}

interface BigramFFConfig {
  type: 'bigram-ff';
  embeddingDim: number;
  numHeads: number;
}

interface GPTConfig {
  type: 'gpt-cpu' | 'gpt-gpu';
  embeddingDim: number;
  numHeads: number;
  numLayers: number;
}

type ModelConfigData = BigramConfig | BigramFFConfig | SingleHeadConfig | MultiHeadConfig | GPTConfig;

interface BaseModelConfigProps<T extends ModelConfigData> {
  isCreating: boolean;
  onConfigChange: (config: T) => void;
}

function BigramModelConfig({ isCreating, onConfigChange }: BaseModelConfigProps<BigramConfig>) {
  const [embeddingDim, setEmbeddingDim] = useState(32);

  const handleEmbeddingDimChange = useCallback((newValue: number) => {
    setEmbeddingDim(newValue);
  }, []);

  // Update config whenever embeddingDim changes
  useEffect(() => {
    onConfigChange({ type: 'bigram', embeddingDim });
  }, [embeddingDim, onConfigChange]);

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
            const value = parseInt(e.target.value);
            if (!isNaN(value)) {
              handleEmbeddingDimChange(value);
            }
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
}

function BigramModelFFConfig({ isCreating, onConfigChange }: BaseModelConfigProps<BigramFFConfig>) {
  const [embeddingDim, setEmbeddingDim] = useState(32);
  const [numHeads, setNumHeads] = useState(2);

  const handleEmbeddingDimChange = useCallback((newValue: number) => {
    setEmbeddingDim(newValue);
  }, []);

  const handleNumHeadsChange = useCallback((newValue: number) => {
    setNumHeads(newValue);
  }, []);

  // Update config whenever values change
  useEffect(() => {
    onConfigChange({ type: 'bigram-ff', embeddingDim, numHeads });
  }, [embeddingDim, numHeads, onConfigChange]);

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
            const value = parseInt(e.target.value);
            if (!isNaN(value)) {
              handleEmbeddingDimChange(value);
            }
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
            const value = parseInt(e.target.value);
            if (!isNaN(value)) {
              handleNumHeadsChange(value);
            }
          }}
          type="number"
          value={numHeads}
        />
      </Label>
      <div className="text-sm text-muted-foreground">
        Bigram model enhanced with multi-head attention and feed-forward layers for improved pattern recognition.
      </div>
    </div>
  );
}

function SingleHeadAttentionConfig({ isCreating, onConfigChange }: BaseModelConfigProps<SingleHeadConfig>) {
  const [embeddingDim, setEmbeddingDim] = useState(32);

  const handleEmbeddingDimChange = useCallback((newValue: number) => {
    setEmbeddingDim(newValue);
  }, []);

  // Update config whenever embeddingDim changes
  useEffect(() => {
    onConfigChange({ type: 'single-head', embeddingDim });
  }, [embeddingDim, onConfigChange]);

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
            const value = parseInt(e.target.value);
            if (!isNaN(value)) {
              handleEmbeddingDimChange(value);
            }
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
}

function MultiHeadAttentionConfig({ isCreating, onConfigChange }: BaseModelConfigProps<MultiHeadConfig>) {
  const [embeddingDim, setEmbeddingDim] = useState(32);
  const [numHeads, setNumHeads] = useState(2);

  const handleEmbeddingDimChange = useCallback((newValue: number) => {
    setEmbeddingDim(newValue);
  }, []);

  const handleNumHeadsChange = useCallback((newValue: number) => {
    setNumHeads(newValue);
  }, []);

  // Update config whenever values change
  useEffect(() => {
    onConfigChange({ type: 'multi-head', embeddingDim, numHeads });
  }, [embeddingDim, numHeads, onConfigChange]);

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
            const value = parseInt(e.target.value);
            if (!isNaN(value)) {
              handleEmbeddingDimChange(value);
            }
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
            const value = parseInt(e.target.value);
            if (!isNaN(value)) {
              handleNumHeadsChange(value);
            }
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

function GPTModelConfig({ isCreating, onConfigChange }: BaseModelConfigProps<GPTConfig>) {
  const [embeddingDim, setEmbeddingDim] = useState(32);
  const [numHeads, setNumHeads] = useState(2);
  const [numLayers, setNumLayers] = useState(2);

  const handleEmbeddingDimChange = useCallback((newValue: number) => {
    setEmbeddingDim(newValue);
  }, []);

  const handleNumHeadsChange = useCallback((newValue: number) => {
    setNumHeads(newValue);
  }, []);

  const handleNumLayersChange = useCallback((newValue: number) => {
    setNumLayers(newValue);
  }, []);

  // Update config whenever values change
  useEffect(() => {
    onConfigChange({ type: 'gpt-cpu', embeddingDim, numHeads, numLayers });
  }, [embeddingDim, numHeads, numLayers, onConfigChange]);

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
            const value = parseInt(e.target.value);
            if (!isNaN(value)) {
              handleEmbeddingDimChange(value);
            }
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
            const value = parseInt(e.target.value);
            if (!isNaN(value)) {
              handleNumHeadsChange(value);
            }
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
            const value = parseInt(e.target.value);
            if (!isNaN(value)) {
              handleNumLayersChange(value);
            }
          }}
          type="number"
          value={numLayers}
        />
      </Label>
      <div className="text-sm text-muted-foreground">
        Full transformer architecture with multiple layers. More layers can learn more complex patterns but are slower
        to train.
      </div>
    </div>
  );
}

type ModelType = 'bigram' | 'bigram-ff' | 'single-head' | 'multi-head' | 'gpt-cpu' | 'gpt-gpu';

export function ModelConfig({
  vocabSize,
  onComplete,
  onBack,
}: {
  onBack: () => void;
  onComplete: (model: LanguageModel | null) => void;
  vocabSize: number;
}) {
  const [modelType, setModelType] = useState<ModelType>('gpt-cpu');
  const [isCreating, setIsCreating] = useState(false);
  const [currentConfig, setCurrentConfig] = useState<ModelConfigData | null>(null);

  const createModel = useCallback(async () => {
    if (isCreating || !currentConfig) return;

    setIsCreating(true);
    try {
      let modelInstance: LanguageModel;

      switch (currentConfig.type) {
        case 'bigram':
          modelInstance = new BigramLanguageModel(vocabSize, currentConfig.embeddingDim, blockSize);
          break;
        case 'bigram-ff':
          modelInstance = new BigramLanguageModelWithFF(
            vocabSize,
            currentConfig.embeddingDim,
            blockSize,
            currentConfig.numHeads,
          );
          break;
        case 'gpt-cpu':
          modelInstance = new GPTModel(
            vocabSize,
            currentConfig.embeddingDim,
            blockSize,
            currentConfig.numHeads,
            currentConfig.numLayers,
          );
          break;
        case 'gpt-gpu': {
          const gpuModel = new GPTModelGPU(
            vocabSize,
            currentConfig.embeddingDim,
            blockSize,
            currentConfig.numHeads,
            currentConfig.numLayers,
          );
          await gpuModel.initializeGPU();
          modelInstance = gpuModel;
          break;
        }
        case 'multi-head':
          modelInstance = new BigramLanguageModelMultiHeadAttention(
            vocabSize,
            currentConfig.embeddingDim,
            blockSize,
            currentConfig.numHeads,
          );
          break;
        case 'single-head':
          modelInstance = new BigramLanguageModelSingleHeadAttention(vocabSize, currentConfig.embeddingDim, blockSize);
          break;
      }

      onComplete(modelInstance);
    } catch (error) {
      console.error('Failed to create model:', error);
      alert(`Failed to create ${currentConfig.type} model: ${errorMessage(error)}`);
    } finally {
      setIsCreating(false);
    }
  }, [isCreating, currentConfig, onComplete, vocabSize]);

  const modelTypes = [
    { value: 'bigram', label: 'Bigram', description: 'Simple bigram model with embeddings' },
    { value: 'bigram-ff', label: 'Bigram with Feed Forward', description: 'Bigram + feed forward layer' },
    { value: 'single-head', label: 'Single Head Attention', description: 'Bigram + single attention head' },
    { value: 'multi-head', label: 'Multi Head Attention', description: 'Bigram + multi-head attention' },
    { value: 'gpt-cpu', label: 'GPT (CPU)', description: 'Full transformer with multiple layers (CPU)' },
    { value: 'gpt-gpu', label: 'GPT (GPU)', description: 'Full transformer with GPU acceleration (WebGPU)' },
  ];

  const handleConfigChange = useCallback((config: ModelConfigData) => {
    setCurrentConfig(config);
  }, []);

  const renderModelParameters = () => {
    let ConfigComponent;
    switch (modelType) {
      case 'bigram':
        ConfigComponent = BigramModelConfig;
        break;
      case 'bigram-ff':
        ConfigComponent = BigramModelFFConfig;
        break;
      case 'gpt-cpu':
      case 'gpt-gpu':
        ConfigComponent = GPTModelConfig;
        break;
      case 'multi-head':
        ConfigComponent = MultiHeadAttentionConfig;
        break;
      case 'single-head':
        ConfigComponent = SingleHeadAttentionConfig;
        break;
    }
    return <ConfigComponent isCreating={isCreating} onConfigChange={handleConfigChange} />;
  };

  return (
    <StepLayout
      backButton={<BackButton disabled={isCreating} onClick={onBack} />}
      completeButton={
        <NextButton
          disabled={isCreating || !currentConfig}
          loading={isCreating}
          onClick={() => void createModel()}
          title={isCreating ? 'Creating model...' : 'Create model'}
        />
      }
      subtitle="Choose the model architecture and configure its parameters"
      title="4. Configure Model"
    >
      <Card className="p-6">
        <div className="space-y-4">
          <div className="space-y-3">
            <Field>
              <FieldLabel>Model Type</FieldLabel>
              <Select
                disabled={isCreating}
                onValueChange={(value) => {
                  setModelType(value as ModelType);
                  onComplete(null);
                  setCurrentConfig(null); // Reset config when changing type
                }}
                value={modelType}
              >
                <SelectTrigger className="cursor-pointer">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {modelTypes.map((type) => (
                    <SelectItem key={type.value} value={type.value}>
                      {type.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>

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
      </Card>
    </StepLayout>
  );
}
