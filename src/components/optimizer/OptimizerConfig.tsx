import { useCallback, useState } from 'react';

import type { Optimizer, Trainable } from '@/llm/types.ts';

import { BackButton } from '@/components/layout/BackButton.tsx';
import { NextButton } from '@/components/layout/NextButton.tsx';
import { StepLayout } from '@/components/layout/StepLayout.tsx';
import { Card } from '@/components/ui/card.tsx';
import { Field, FieldLabel } from '@/components/ui/field.tsx';
import { Input } from '@/components/ui/input.tsx';
import { Label } from '@/components/ui/label.tsx';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.tsx';
import { AdamWOptimizer } from '@/llm/optimizers/AdamWOptimizer.ts';
import { SDGOptimizer } from '@/llm/optimizers/SDGOptimizer.ts';

type OptimizerType = 'adamw' | 'sgd';

export function OptimizerConfig({
  model,
  onComplete,
  onBack,
}: {
  model: Trainable;
  onBack: () => void;
  onComplete: (optimizer: Optimizer) => void;
}) {
  const [optimizerType, setOptimizerType] = useState<OptimizerType>('adamw');

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
      optimizerInstance = new AdamWOptimizer(model, learningRate, beta1, beta2, epsilon, weightDecay);
    } else {
      optimizerInstance = new SDGOptimizer(model, learningRate);
    }

    onComplete(optimizerInstance);
  }, [optimizerType, onComplete, model, learningRate, beta1, beta2, epsilon, weightDecay]);

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
                const value = parseFloat(e.target.value);
                if (!isNaN(value)) {
                  setLearningRate(value);
                }
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
                  const value = parseFloat(e.target.value);
                  if (!isNaN(value)) {
                    setBeta1(value);
                  }
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
                  const value = parseFloat(e.target.value);
                  if (!isNaN(value)) {
                    setBeta2(value);
                  }
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
                  const value = parseFloat(e.target.value);
                  if (!isNaN(value)) {
                    setEpsilon(value);
                  }
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
                  const value = parseFloat(e.target.value);
                  if (!isNaN(value)) {
                    setWeightDecay(value);
                  }
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
                const value = parseFloat(e.target.value);
                if (!isNaN(value)) {
                  setLearningRate(value);
                }
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
    <StepLayout
      backButton={<BackButton onClick={onBack} />}
      completeButton={<NextButton onClick={createOptimizer} title="Create optimizer" />}
      subtitle="Choose the optimizer algorithm and configure its parameters"
      title="5. Configure optimizer"
    >
      <Card className="p-6">
        <div className="space-y-4">
          <div className="space-y-3">
            <Field>
              <FieldLabel>Optimizer Type</FieldLabel>
              <Select
                onValueChange={(value) => {
                  setOptimizerType(value as OptimizerType);
                }}
                value={optimizerType}
              >
                <SelectTrigger className="cursor-pointer">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {optimizerTypes.map((type) => (
                    <SelectItem key={type.value} value={type.value}>
                      {type.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>

            <div className="text-sm text-muted-foreground">
              {optimizerTypes.find((t) => t.value === optimizerType)?.description}
            </div>
          </div>

          {renderOptimizerParameters()}
        </div>
      </Card>
    </StepLayout>
  );
}
