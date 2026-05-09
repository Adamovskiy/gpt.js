import type { Trainable } from '../../llm/types.ts';
import { Button } from '../ui/button.tsx';
import { type Optimizer, UniversalAdamWOptimizer, GPUAdamWOptimizer } from '../../llm/optimizers/optimizers.ts';
import { useEffect } from 'react';

export function OptimizerConfig({
  model,
  optimizer,
  setOptimizer,
  onOptimizerChange,
}: {
  model: Trainable;
  optimizer: Optimizer | undefined;
  setOptimizer: (optimizer: Optimizer | undefined) => void;
  onOptimizerChange?: () => void;
}) {
  useEffect(() => {
    setOptimizer(undefined);
    onOptimizerChange?.(); // Clear chart when model changes
  }, [model, setOptimizer, onOptimizerChange]);

  const isGPUModel = (model as { isGPU?: boolean })?.isGPU === true;

  return (
    <div>
      {optimizer && <div>{(optimizer as object).constructor.name}</div>}
      <div className="space-y-2">
        <Button
          onClick={() => {
            const optimizer: Optimizer = new UniversalAdamWOptimizer(model, 3e-4, 0.9, 0.999, 1e-8, 0.01);
            setOptimizer(optimizer);
            onOptimizerChange?.(); // Clear chart when optimizer changes
          }}
        >
          {optimizer ? 'Re-set' : 'Set'} CPU AdamW optimizer
        </Button>
      </div>
    </div>
  );
}
