import { ChevronLeft, ChevronRight } from 'lucide-react';
import { useCallback, useState } from 'react';

import type { Trainable } from '../../llm/types.ts';

import { SDGOptimizer } from '../../llm/optimizers/SDGOptimizer.ts';
import { UniversalAdamWOptimizer } from '../../llm/optimizers/UniversalAdamWOptimizer.ts';
import { type Optimizer } from '../../llm/optimizers/utils.ts';
import { Button } from '../ui/button.tsx';
import { Card } from '../ui/card.tsx';
import { Input } from '../ui/input.tsx';
import { Label } from '../ui/label.tsx';

export interface OptimizerConfigProps {
  model: Trainable;
  onComplete: (optimizer: Optimizer) => void;
  onBack: () => void;
}

export function OptimizerConfig({ model, onComplete, onBack }: OptimizerConfigProps) {
  const [optimizerType, setOptimizerType] = useState<string>('adamw');
  const [optimizer, setOptimizer] = useState<Optimizer | undefined>();

  // Optimizer parameters with defaults
  const [learningRate, setLearningRate] = useState(3e-4);
  const [beta1, setBeta1] = useState(0.9);
  const [beta2, setBeta2] = useState(0.999);
  const [epsilon, setEpsilon] = useState(1e-8);
  const [weightDecay, setWeightDecay] = useState(0.01);

  const optimizerTypes = [
    { value: 'adamw', label: 'AdamW', description: 'Adam with weight decay - good default choice' },
    { value: 'sgd', label: 'SGD', description: 'Stochastic Gradient Descent - simple but effective' },
  ];

  const createOptimizer = useCallback(() => {
    let optimizerInstance: Optimizer;

    if (optimizerType === 'adamw') {
      optimizerInstance = new UniversalAdamWOptimizer(model, learningRate, beta1, beta2, epsilon, weightDecay);
    } else {
      optimizerInstance = new SDGOptimizer(model, learningRate);
    }

    setOptimizer(optimizerInstance);
  }, [model, optimizerType, learningRate, beta1, beta2, epsilon, weightDecay]);

  const handleFinish = useCallback(() => {
    if (optimizer) {
      onComplete(optimizer);
    }
  }, [optimizer, onComplete]);

  const renderOptimizerParameters = () => {
    if (optimizerType === 'adamw') {
      return (
        <div className="space-y-3">
          <Label>
            Learning Rate
            <Input
              className="mt-1"
              max="0.1"
              min="0.0001"
              onChange={(e) => {
                setLearningRate(parseFloat(e.target.value) || 3e-4);
              }}
              step="0.0001"
              type="number"
              value={learningRate}
            />
          </Label>
          <div className="grid grid-cols-2 gap-3">
            <Label>
              Beta 1
              <Input
                className="mt-1"
                max="0.999"
                min="0.1"
                onChange={(e) => {
                  setBeta1(parseFloat(e.target.value) || 0.9);
                }}
                step="0.01"
                type="number"
                value={beta1}
              />
            </Label>
            <Label>
              Beta 2
              <Input
                className="mt-1"
                max="0.999"
                min="0.1"
                onChange={(e) => {
                  setBeta2(parseFloat(e.target.value) || 0.999);
                }}
                step="0.01"
                type="number"
                value={beta2}
              />
            </Label>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Label>
              Epsilon
              <Input
                className="mt-1"
                max="1e-6"
                min="1e-10"
                onChange={(e) => {
                  setEpsilon(parseFloat(e.target.value) || 1e-8);
                }}
                step="0.0000001"
                type="number"
                value={epsilon}
              />
            </Label>
            <Label>
              Weight Decay
              <Input
                className="mt-1"
                max="0.1"
                min="0"
                onChange={(e) => {
                  setWeightDecay(parseFloat(e.target.value) || 0.01);
                }}
                step="0.001"
                type="number"
                value={weightDecay}
              />
            </Label>
          </div>
          <div className="text-sm text-muted-foreground">
            AdamW is an adaptive optimizer with momentum and weight decay. Default values work well for most cases.
          </div>
        </div>
      );
    } else {
      return (
        <div className="space-y-3">
          <Label>
            Learning Rate
            <Input
              className="mt-1"
              max="1.0"
              min="0.001"
              onChange={(e) => {
                setLearningRate(parseFloat(e.target.value) || 0.01);
              }}
              step="0.001"
              type="number"
              value={learningRate}
            />
          </Label>
          <div className="text-sm text-muted-foreground">
            SGD is a simple optimizer that updates parameters using gradients. Typically requires higher learning rates.
          </div>
        </div>
      );
    }
  };

  return (
    <Card className="p-6">
      <div className="space-y-6">
        <div>
          <h3 className="text-lg font-semibold">Create Optimizer</h3>
          <p className="text-sm text-muted-foreground">Choose the optimizer algorithm and configure its parameters</p>
        </div>

        <div className="space-y-4">
          <div className="space-y-3">
            <Label htmlFor="optimizer-type-select">Optimizer Type</Label>
            <select
              className="
                w-full rounded-md border border-input bg-background p-2
              "
              id="optimizer-type-select"
              onChange={(e) => {
                setOptimizerType(e.target.value);
              }}
              value={optimizerType}
            >
              {optimizerTypes.map((type) => (
                <option key={type.value} value={type.value}>
                  {type.label}
                </option>
              ))}
            </select>

            <div className="text-sm text-muted-foreground">
              {optimizerTypes.find((t) => t.value === optimizerType)?.description}
            </div>
          </div>

          {renderOptimizerParameters()}
        </div>

        {optimizer && (
          <Card className="bg-green-50 p-4">
            <div className="space-y-2">
              <div className="text-sm font-medium">
                {(optimizer as { constructor: { name: string } }).constructor.name} created
              </div>
              <div className="text-xs text-muted-foreground">Ready to train with learning rate: {learningRate}</div>
            </div>
          </Card>
        )}

        <div className="flex justify-between">
          <Button onClick={onBack} variant="outline">
            <ChevronLeft className="mr-1 size-4" />
            Back
          </Button>

          {!optimizer ? (
            <Button onClick={createOptimizer}>Create Optimizer</Button>
          ) : (
            <Button onClick={handleFinish}>
              Next: Train & Generate
              <ChevronRight className="ml-1 size-4" />
            </Button>
          )}
        </div>
      </div>
    </Card>
  );
}
