import type { LanguageModel } from '../../types.ts';
import { GPTModel } from '../../tfOps.ts';
import { blockSize } from '../../sampling.ts';
import { Button } from '../ui/button.tsx';
import { Badge } from '../ui/badge.tsx';

export function ModelConfig({
  vocabSize,
  model,
  setModel,
}: {
  vocabSize: number;
  model: LanguageModel | undefined;
  setModel: (model: LanguageModel) => void;
}) {
  return (
    <div>
      {model && (
        <div>
          {(model as object).constructor.name}
          <Badge variant="outline">{model.getParameters().length} parameters</Badge>
        </div>
      )}
      <Button
        onClick={() => {
          const numberEmbeddingDimensions = 32;
          const numHeads = 2; // Reduce heads
          const numLayers = 2; // Reduce layers
          const model: LanguageModel = new GPTModel(
            vocabSize,
            numberEmbeddingDimensions,
            blockSize,
            numHeads,
            numLayers,
          );
          setModel(model);
        }}
      >
        {model ? 'Re-set' : 'Set'} model
      </Button>
    </div>
  );
}
