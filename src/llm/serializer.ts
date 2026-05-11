import type { LanguageModel, Optimizer, Tokenizer } from '@/llm/types.ts';

import { BigramLanguageModel, type BigramLanguageModelSerializedData } from '@/llm/models/BigramLanguageModel.ts';
import {
  BigramLanguageModelMultiHeadAttention,
  type BigramLanguageModelMultiHeadAttentionSerializedData,
} from '@/llm/models/BigramLanguageModelMultiHeadAttention.ts';
import {
  BigramLanguageModelSingleHeadAttention,
  type BigramLanguageModelSingleHeadAttentionSerializedData,
} from '@/llm/models/BigramLanguageModelSingleHeadAttention.ts';
import {
  BigramLanguageModelWithFF,
  type BigramLanguageModelWithFFSerializedData,
} from '@/llm/models/BigramLanguageModelWithFF.ts';
import { GPTModel, type GPTModelSerializedData } from '@/llm/models/GPTModel.ts';
import { GPTModelGPU } from '@/llm/models/gpu/GPTModelGPU.ts';
import { AdamWOptimizer, type AdamWOptimizerSerializedData } from '@/llm/optimizers/AdamWOptimizer.ts';
import { SDGOptimizer, type SDGOptimizerSerializedData } from '@/llm/optimizers/SDGOptimizer.ts';
import { BPETokenizer, type BPETokenizerSerializedData } from '@/llm/tokenizers/BPETokenizer.ts';
import { CharTokenizer, type CharTokenizerSerializedData } from '@/llm/tokenizers/CharTokenizer.ts';

export function serializeModel(model: LanguageModel) {
  if (model instanceof GPTModel)
    return {
      type: 'GPTModel' as const,
      serializedData: model.getSerializedData(),
    };
  if (model instanceof BigramLanguageModel)
    return {
      type: 'BigramLanguageModel' as const,
      serializedData: model.getSerializedData(),
    };
  if (model instanceof BigramLanguageModelWithFF)
    return {
      type: 'BigramLanguageModelWithFF' as const,
      serializedData: model.getSerializedData(),
    };
  if (model instanceof BigramLanguageModelSingleHeadAttention)
    return {
      type: 'BigramLanguageModelSingleHeadAttention' as const,
      serializedData: model.getSerializedData(),
    };
  if (model instanceof BigramLanguageModelMultiHeadAttention)
    return {
      type: 'BigramLanguageModelMultiHeadAttention' as const,
      serializedData: model.getSerializedData(),
    };
  if (model instanceof GPTModelGPU)
    return {
      type: 'GPTModelGPU' as const,
      serializedData: model.getSerializedData(),
    };
  throw new Error(`Unsupported model type: ${model.constructor.name}`);
}

export function serializeOptimizer(optimizer: Optimizer) {
  if (optimizer instanceof AdamWOptimizer)
    return {
      type: 'AdamWOptimizer' as const,
      serializedData: optimizer.getSerializedData(),
    };
  if (optimizer instanceof SDGOptimizer)
    return {
      type: 'SDGOptimizer' as const,
      serializedData: optimizer.getSerializedData(),
    };
  throw new Error(`Unsupported optimizer type: ${optimizer.constructor.name}`);
}

export function serializeTokenizer(tokenizer: Tokenizer) {
  if (tokenizer instanceof BPETokenizer)
    return {
      type: 'BPETokenizer' as const,
      serializedData: tokenizer.getSerializedData(),
    };
  if (tokenizer instanceof CharTokenizer)
    return {
      type: 'CharTokenizer' as const,
      serializedData: tokenizer.getSerializedData(),
    };
  throw new Error(`Unsupported optimizer type: ${tokenizer.constructor.name}`);
}

export type ModelData =
  | {
      serializedData: GPTModelSerializedData;
      type: 'GPTModel';
    }
  | {
      serializedData: BigramLanguageModelSerializedData;
      type: 'BigramLanguageModel';
    }
  | {
      serializedData: BigramLanguageModelWithFFSerializedData;
      type: 'BigramLanguageModelWithFF';
    }
  | {
      serializedData: BigramLanguageModelSingleHeadAttentionSerializedData;
      type: 'BigramLanguageModelSingleHeadAttention';
    }
  | {
      serializedData: BigramLanguageModelMultiHeadAttentionSerializedData;
      type: 'BigramLanguageModelMultiHeadAttention';
    }
  | { serializedData: unknown; type: 'GPTModelGPU' };

export type OptimizerData =
  | {
      serializedData: AdamWOptimizerSerializedData;
      type: 'AdamWOptimizer';
    }
  | {
      serializedData: SDGOptimizerSerializedData;
      type: 'SDGOptimizer';
    };

export type TokenizerData =
  | {
      serializedData: BPETokenizerSerializedData;
      type: 'BPETokenizer';
    }
  | {
      serializedData: CharTokenizerSerializedData;
      type: 'CharTokenizer';
    };

export async function deserializeModel(modelData: ModelData) {
  switch (modelData.type) {
    case 'BigramLanguageModel': {
      const { BigramLanguageModel } = await import('@/llm/models/BigramLanguageModel.ts');
      return BigramLanguageModel.fromSerializedData(modelData.serializedData);
    }
    case 'BigramLanguageModelMultiHeadAttention': {
      const { BigramLanguageModelMultiHeadAttention } =
        await import('@/llm/models/BigramLanguageModelMultiHeadAttention.ts');
      return BigramLanguageModelMultiHeadAttention.fromSerializedData(modelData.serializedData);
    }
    case 'BigramLanguageModelSingleHeadAttention': {
      const { BigramLanguageModelSingleHeadAttention } =
        await import('@/llm/models/BigramLanguageModelSingleHeadAttention.ts');
      return BigramLanguageModelSingleHeadAttention.fromSerializedData(modelData.serializedData);
    }
    case 'BigramLanguageModelWithFF': {
      const { BigramLanguageModelWithFF } = await import('@/llm/models/BigramLanguageModelWithFF.ts');
      return BigramLanguageModelWithFF.fromSerializedData(modelData.serializedData);
    }
    case 'GPTModel': {
      const { GPTModel } = await import('@/llm/models/GPTModel.ts');
      return GPTModel.fromSerializedData(modelData.serializedData);
    }
    case 'GPTModelGPU': {
      const { GPTModelGPU } = await import('@/llm/models/gpu/GPTModelGPU.ts');
      return GPTModelGPU.fromSerializedData(modelData.serializedData);
    }
  }
}

export async function deserializeOptimizer(model: LanguageModel, optimizerData: OptimizerData) {
  switch (optimizerData.type) {
    case 'AdamWOptimizer': {
      const { AdamWOptimizer } = await import('@/llm/optimizers/AdamWOptimizer.ts');
      return AdamWOptimizer.fromSerializedData(optimizerData.serializedData, model);
    }
    case 'SDGOptimizer': {
      const { SDGOptimizer } = await import('@/llm/optimizers/SDGOptimizer.ts');
      return SDGOptimizer.fromSerializedData(optimizerData.serializedData, model);
    }
  }
}

export async function deserializeTokenizer(tokenizerData: TokenizerData) {
  switch (tokenizerData.type) {
    case 'BPETokenizer': {
      const { BPETokenizer } = await import('@/llm/tokenizers/BPETokenizer.ts');
      return BPETokenizer.fromSerializedData(tokenizerData.serializedData);
    }
    case 'CharTokenizer': {
      const { CharTokenizer } = await import('@/llm/tokenizers/CharTokenizer.ts');
      return CharTokenizer.fromSerializedData(tokenizerData.serializedData);
    }
  }
}
