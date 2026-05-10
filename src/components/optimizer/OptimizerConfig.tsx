import type { Trainable } from '../../llm/types.ts';
import { Button } from '../ui/button.tsx';
import { Card } from '../ui/card.tsx';
import { Input } from '../ui/input.tsx';
import { Label } from '../ui/label.tsx';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { type Optimizer } from '../../llm/optimizers/utils.ts';
import { useState, useCallback } from 'react';
import { SDGOptimizer } from '../../llm/optimizers/SDGOptimizer.ts';
import { UniversalAdamWOptimizer } from '../../llm/optimizers/UniversalAdamWOptimizer.ts';

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
              type="number"
              step="0.0001"
              min="0.0001"
              max="0.1"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value) || 3e-4)}
              className="mt-1"
            />
          </Label>
          <div className="grid grid-cols-2 gap-3">
            <Label>
              Beta 1
              <Input
                type="number"
                step="0.01"
                min="0.1"
                max="0.999"
                value={beta1}
                onChange={(e) => setBeta1(parseFloat(e.target.value) || 0.9)}
                className="mt-1"
              />
            </Label>
            <Label>
              Beta 2
              <Input
                type="number"
                step="0.01"
                min="0.1"
                max="0.999"
                value={beta2}
                onChange={(e) => setBeta2(parseFloat(e.target.value) || 0.999)}
                className="mt-1"
              />
            </Label>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Label>
              Epsilon
              <Input
                type="number"
                step="0.0000001"
                min="1e-10"
                max="1e-6"
                value={epsilon}
                onChange={(e) => setEpsilon(parseFloat(e.target.value) || 1e-8)}
                className="mt-1"
              />
            </Label>
            <Label>
              Weight Decay
              <Input
                type="number"
                step="0.001"
                min="0"
                max="0.1"
                value={weightDecay}
                onChange={(e) => setWeightDecay(parseFloat(e.target.value) || 0.01)}
                className="mt-1"
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
              type="number"
              step="0.001"
              min="0.001"
              max="1.0"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.01)}
              className="mt-1"
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
              id="optimizer-type-select"
              value={optimizerType}
              onChange={(e) => setOptimizerType(e.target.value)}
              className="w-full p-2 border border-input rounded-md bg-background"
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
          <Card className="p-4 bg-green-50">
            <div className="space-y-2">
              <div className="text-sm font-medium">
                {(optimizer as { constructor: { name: string } }).constructor.name} created
              </div>
              <div className="text-xs text-muted-foreground">Ready to train with learning rate: {learningRate}</div>
            </div>
          </Card>
        )}

        <div className="flex justify-between">
          <Button variant="outline" onClick={onBack}>
            <ChevronLeft className="w-4 h-4 mr-1" />
            Back
          </Button>

          {!optimizer ? (
            <Button onClick={createOptimizer}>Create Optimizer</Button>
          ) : (
            <Button onClick={handleFinish}>
              Next: Train & Generate
              <ChevronRight className="w-4 h-4 ml-1" />
            </Button>
          )}
        </div>
      </div>
    </Card>
  );
}
