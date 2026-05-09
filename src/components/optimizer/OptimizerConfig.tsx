import type { Trainable } from '../../types.ts';
import { Button } from '../ui/button.tsx';
import { type Optimizer, UniversalAdamWOptimizer } from '../../optimizers.ts';
import { useEffect } from 'react';

export function OptimizerConfig({
  model,
  optimizer,
  setOptimizer,
}: {
  model: Trainable;
  optimizer: Optimizer | undefined;
  setOptimizer: (optimizer: Optimizer | undefined) => void;
}) {
  useEffect(() => setOptimizer(undefined), [model]);

  return (
    <div>
      {optimizer && <div>{(optimizer as object).constructor.name}</div>}
      <Button
        onClick={() => {
          const optimizer: Optimizer = new UniversalAdamWOptimizer(model, 3e-4, 0.9, 0.999, 1e-8, 0.01);
          setOptimizer(optimizer);
        }}
      >
        {optimizer ? 'Re-set' : 'Set'} optimizer
      </Button>
    </div>
  );
}
