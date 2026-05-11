# GPT.js - Interactive GPT Implementation in TypeScript

> *"If you can't explain something to a first year student, then you haven't really understood."*  
> – Richard Feynman

## Overview

GPT.js is an educational implementation of a GPT (Generative Pre-trained Transformer) model built entirely in TypeScript. This project demonstrates the core concepts of language models and neural networks in an interactive, browser-based environment.

**🎯 Goal**: Make machine learning accessible to developers who have no data science background and prefer TypeScript/JavaScript over Python.

**🚨 Important**: This is an educational tool, not intended for production use. The focus is on clarity and understanding rather than performance.

## Live Demo

**🌐 [Try it live: nanogpt.adamovskiy.com](https://nanogpt.adamovskiy.com)**

## Features

- **Interactive Learning**: Step-by-step walkthrough of GPT implementation
- **Visual Introspection**: Explore each component's internals and data flow
- **Dual Implementation**: CPU (readable) and GPU (performant) versions
- **Custom Training Data**: Upload your own text files for training
- **Real-time Visualization**: Watch the model learn and generate text

## Inspiration

This project is based on [Andrej Karpathy's NanoGPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) with several enhancements:

- **Dynamic Introspection**: Every component can be inspected at runtime
- **Verbose Implementation**: Clear variable names and extensive documentation
- **GPU Acceleration**: WebGL implementation for practical training speeds
- **Educational Focus**: All math operations implemented from scratch (no black-box libraries like TensorFlow)

## Architecture

The application is structured around four main components:

### 1. **Input Component**
- Upload and manage training/validation datasets
- Supports text files of various sizes
- Default dataset: Tiny Shakespeare or War and Peace by Tolstoy

### 2. **Tokenizer**
- Converts text sequences to numerical tokens
- Supports two algorithms:
  - **Character-level**: Simple one-to-one character mapping
  - **BPE (Byte Pair Encoding)**: Learns common character pairs for efficiency

### 3. **Model**
- Mathematical representation of language patterns
- Transformer architecture with attention mechanisms
- Configurable parameters (layers, heads, embedding dimensions)

### 4. **Optimizer**
- Algorithms for adjusting model parameters during training
- Implementations include SGD and AdamW
- Visualizes loss reduction over time

## Project Structure

```
src/
├── components/     # React UI components for interactive exploration
├── gpu/           # WebGL-based GPU implementations for performance
├── data/          # Training datasets
├── lib/           # Utility functions and helpers
├── llm/           # Core CPU implementations of ML algorithms
└── workers/       # Web Workers for non-blocking computations
```

## Getting Started

### Prerequisites
- Node.js 18+ and npm
- Modern browser with WebGL support

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd gptjs

# Install dependencies
npm install

# Start development server
npm run dev
```

### Usage

1. **Upload Training Data**: Choose a text file or use the default dataset
2. **Configure Tokenizer**: Select between Character or BPE tokenization
3. **Design Model**: Set architecture parameters (layers, attention heads, etc.)
4. **Train Model**: Watch the model learn patterns in your data
5. **Generate Text**: Use the trained model to generate new text

## Educational Value

This project helps you understand:

- **Tokenization**: How text becomes numbers
- **Embeddings**: How words become vectors
- **Attention Mechanisms**: How models focus on relevant parts
- **Transformer Architecture**: The building blocks of modern LLMs
- **Training Process**: How models learn from data
- **Text Generation**: How models produce coherent text

## Technical Implementation

### Type Safety
- Fully typed TypeScript with strict configuration
- Generic interfaces for extensibility
- No `any` types or unsafe casts

### Performance
- Web Workers prevent UI blocking during training
- WebGL shaders for GPU-accelerated computations
- Efficient tensor operations with proper memory management

### Code Quality
- Comprehensive linting with ESLint
- Formatted with Prettier
- Component-based architecture with clear separation of concerns

## Roadmap & TODO
- [ ] Fix linter and type errors
- [ ] Fix UX of model usage page: persist stats, limit chart's RAM usage, abort training, etc
- [ ] Implement serialization of all models
- [ ] Tests and edge case handling
- [ ] Overfitting analysis
- [ ] GPU implementation of AdamW optimizer
- [ ] Visualization of getBatch function
- [ ] Model parameter introspection tools
- [ ] Optimizer operation visualization
- [ ] Tensor operations visualization from `tensorOps.ts`
- [ ] Improvement of model configuration steps UX
- [ ] Make blockSize and batchSize configurable
- [ ] Stream inference output token by token

## Contributing

Please read [CLAUDE.md](./CLAUDE.md) for development guidelines and coding standards.

## Acknowledgments

- **[Andrej Karpathy](https://github.com/karpathy)** for the original [NanoGPT](https://github.com/karpathy/nanoGPT) implementation and excellent educational content
- **OpenAI** for the transformer architecture and attention mechanisms
- **The broader ML community** for making this knowledge accessible