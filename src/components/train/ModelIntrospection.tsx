import { useState } from 'react';

import type { Tensor1d, Tensor2d } from '@/llm/tensorOps.ts';
import type { LanguageModel } from '@/llm/types.ts';

import { Button } from '@/components/ui/button.tsx';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.tsx';

interface ParameterInfo {
  name: string;
  shape: string;
  size: number;
  data: Tensor1d | Tensor2d;
}

function formatTensorShape(tensor: Tensor1d | Tensor2d): string {
  if (Array.isArray(tensor[0])) {
    const tensor2d = tensor as Tensor2d;
    return `${tensor2d.length}x${tensor2d[0].length}`;
  }
  return `${(tensor as Tensor1d).length}`;
}

function getTensorSize(tensor: Tensor1d | Tensor2d): number {
  if (Array.isArray(tensor[0])) {
    const tensor2d = tensor as Tensor2d;
    return tensor2d.length * tensor2d[0].length;
  }
  return (tensor as Tensor1d).length;
}

function formatNumber(num: number): string {
  if (Math.abs(num) < 0.0001 && num !== 0) {
    return num.toExponential(3);
  }
  return num.toFixed(4);
}

function TensorViewer({ data, name }: { data: Tensor1d | Tensor2d; name: string }) {
  const [isExpanded, setIsExpanded] = useState(false);

  const renderTensorContent = () => {
    if (Array.isArray(data[0])) {
      const tensor2d = data as Tensor2d;
      return (
        <div className="max-h-96 overflow-auto">
          <table className="w-max border-separate border-spacing-1 font-mono text-xs">
            <thead>
              <tr>
                <th className="sticky left-0 bg-background px-2 py-1 text-muted-foreground"></th>
                {tensor2d[0].map((_, colIndex) => (
                  <th className="px-2 py-1 text-muted-foreground" key={colIndex}>
                    [{colIndex}]
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tensor2d.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  <td className="sticky left-0 bg-background px-2 py-1 text-muted-foreground">[{rowIndex}]</td>
                  {row.map((val, colIndex) => (
                    <td className="px-2 py-1 text-right" key={colIndex}>
                      {formatNumber(val)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    } else {
      const tensor1d = data as Tensor1d;
      return (
        <div className="max-h-96 overflow-auto">
          <div className="font-mono text-xs">
            <div className="grid w-max auto-cols-max grid-flow-col gap-2">
              {tensor1d.map((val, index) => (
                <div className="text-center" key={index}>
                  <div className="text-muted-foreground">[{index}]</div>
                  <div className="border px-2 py-1">{formatNumber(val)}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      );
    }
  };

  return (
    <div className="border-l-2 border-border pl-4">
      <Button
        className="h-auto w-full justify-start p-2 text-left"
        onClick={() => {
          setIsExpanded(!isExpanded);
        }}
        variant="ghost"
      >
        <span className="mr-2">{isExpanded ? '▼' : '▶'}</span>
        <span className="font-medium">{name}</span>
        <span className="ml-2 text-sm text-muted-foreground">
          ({formatTensorShape(data)}) - {getTensorSize(data)} parameters
        </span>
      </Button>
      {isExpanded && (
        <div className="mt-2 pr-4 pb-4 pl-6">
          <div className="w-0 min-w-full rounded-sm border bg-muted/30 p-3">{renderTensorContent()}</div>
        </div>
      )}
    </div>
  );
}

function ParameterGroup({ groupName, parameters }: { groupName: string; parameters: ParameterInfo[] }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const totalParams = parameters.reduce((sum, param) => sum + param.size, 0);

  return (
    <Card>
      <CardHeader className="pb-3">
        <Button
          className="h-auto w-full justify-start p-0 text-left"
          onClick={() => {
            setIsExpanded(!isExpanded);
          }}
          variant="ghost"
        >
          <span className="mr-2">{isExpanded ? '▼' : '▶'}</span>
          <div>
            <CardTitle className="text-left">{groupName}</CardTitle>
            <CardDescription>
              {parameters.length} field{parameters.length !== 1 ? 's' : ''}, {totalParams.toLocaleString()} parameters
            </CardDescription>
          </div>
        </Button>
        {isExpanded && (
          <CardContent className="pt-4">
            <div className="space-y-2">
              {parameters.map((param) => (
                <TensorViewer data={param.data} key={param.name} name={param.name} />
              ))}
            </div>
          </CardContent>
        )}
      </CardHeader>
    </Card>
  );
}

function modelLabel(className: string) {
  switch (className) {
    case 'BigramLanguageModel':
      return 'Bigram';
    case 'BigramLanguageModelMultiHeadAttention':
      return 'Single Head Attention';
    case 'BigramLanguageModelSingleHeadAttention':
      return 'Multi Head Attention';
    case 'BigramLanguageModelWithFF':
      return 'Bigram with Feed Forward';
    case 'GPTModel':
      return 'GPT (CPU)';
    case 'GPTModelGPU':
      return 'GPT (GPU)';
    default:
      return className;
  }
}

export function ModelIntrospection({ model }: { model: LanguageModel }) {
  const parameters = model.getParameters();
  const totalParams = parameters.reduce((sum, param) => sum + getTensorSize(param.data), 0);

  // Group parameters by their prefix (e.g., "layer0_", "lnFinal_", etc.)
  const groupedParams: Record<string, ParameterInfo[]> = {};

  parameters.forEach((param) => {
    const paramInfo: ParameterInfo = {
      name: param.name,
      shape: formatTensorShape(param.data),
      size: getTensorSize(param.data),
      data: param.data,
    };

    // Use actual parameter name as group - just take the first part before underscore if it exists
    const groupName = param.name.includes('_') ? param.name.split('_')[0] : param.name;

    groupedParams[groupName] ??= [];
    groupedParams[groupName].push(paramInfo);
  });

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Model Overview</CardTitle>
          <CardDescription>Total parameters: {totalParams.toLocaleString()}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 text-sm md:grid-cols-4">
            <div>
              <div className="font-medium">Parameter Groups</div>
              <div className="text-muted-foreground">{Object.keys(groupedParams).length}</div>
            </div>
            <div>
              <div className="font-medium">Individual Parameters</div>
              <div className="text-muted-foreground">{parameters.length}</div>
            </div>
            <div>
              <div className="font-medium">Model Type</div>
              <div className="text-muted-foreground">{modelLabel(model.constructor.name)}</div>
            </div>
            <div>
              <div className="font-medium">Memory (approx)</div>
              <div className="text-muted-foreground">{((totalParams * 4) / 1024 / 1024).toFixed(1)} MB</div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="space-y-4">
        {Object.entries(groupedParams)
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([groupName, params]) => (
            <ParameterGroup groupName={groupName} key={groupName} parameters={params} />
          ))}
      </div>
    </div>
  );
}
