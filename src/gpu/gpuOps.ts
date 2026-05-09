import matmulShader from './shaders/matmul.wgsl?raw';
import softmaxShader from './shaders/softmax.wgsl?raw';
import elementwiseShader from './shaders/elementwise.wgsl?raw';
import layernormShader from './shaders/layernorm.wgsl?raw';
import { type Tensor2d, type Tensor3d } from '../llm/tensorOps.ts';

export class GPUOperations {
  private device: GPUDevice;
  private matmulPipeline: GPUComputePipeline | null = null;
  private softmaxPipelines: {
    findMax: GPUComputePipeline | null;
    computeExpSum: GPUComputePipeline | null;
    normalize: GPUComputePipeline | null;
  } = { findMax: null, computeExpSum: null, normalize: null };
  private elementwisePipeline: GPUComputePipeline | null = null;
  private layernormPipelines: {
    computeMeans: GPUComputePipeline | null;
    computeVariances: GPUComputePipeline | null;
    normalize: GPUComputePipeline | null;
  } = { computeMeans: null, computeVariances: null, normalize: null };

  constructor(device: GPUDevice) {
    this.device = device;
  }

  async initializePipelines(): Promise<void> {
    // Matrix multiplication pipeline
    const matmulModule = this.device.createShaderModule({ code: matmulShader });
    this.matmulPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: matmulModule, entryPoint: 'main' },
    });

    // Softmax pipelines
    const softmaxModule = this.device.createShaderModule({ code: softmaxShader });
    this.softmaxPipelines.findMax = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: softmaxModule, entryPoint: 'findMax' },
    });
    this.softmaxPipelines.computeExpSum = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: softmaxModule, entryPoint: 'computeExpSum' },
    });
    this.softmaxPipelines.normalize = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: softmaxModule, entryPoint: 'normalize' },
    });

    // Element-wise operations pipeline
    const elementwiseModule = this.device.createShaderModule({ code: elementwiseShader });
    this.elementwisePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: elementwiseModule, entryPoint: 'main' },
    });

    // LayerNorm pipelines
    const layernormModule = this.device.createShaderModule({ code: layernormShader });
    this.layernormPipelines.computeMeans = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: layernormModule, entryPoint: 'computeMeans' },
    });
    this.layernormPipelines.computeVariances = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: layernormModule, entryPoint: 'computeVariances' },
    });
    this.layernormPipelines.normalize = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: layernormModule, entryPoint: 'normalize' },
    });
  }

  private createBuffer(data: Float32Array, usage: GPUBufferUsageFlags): GPUBuffer {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: usage | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  private async readBuffer(buffer: GPUBuffer, size: number): Promise<Float32Array> {
    const readBuffer = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, size);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    readBuffer.destroy();

    return result;
  }

  async matrixMultiply(a: Tensor2d, b: Tensor2d): Promise<Tensor2d> {
    if (!this.matmulPipeline) {
      throw new Error('Matrix multiplication pipeline not initialized');
    }

    const M = a.length;
    const K = a[0].length;
    const N = b[0].length;

    // Flatten matrices
    const aFlat = new Float32Array(a.flat());
    const bFlat = new Float32Array(b.flat());
    const resultSize = M * N;

    // Create buffers
    const paramsBuffer = this.device.createBuffer({
      size: 16, // 3 * 4 bytes + padding
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([M, N, K]));

    const aBuffer = this.createBuffer(aFlat, GPUBufferUsage.STORAGE);
    const bBuffer = this.createBuffer(bFlat, GPUBufferUsage.STORAGE);
    const resultBuffer = this.device.createBuffer({
      size: resultSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: this.matmulPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: aBuffer } },
        { binding: 2, resource: { buffer: bBuffer } },
        { binding: 3, resource: { buffer: resultBuffer } },
      ],
    });

    // Execute computation
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.matmulPipeline);
    passEncoder.setBindGroup(0, bindGroup);

    const workgroupsX = Math.ceil(M / 16);
    const workgroupsY = Math.ceil(N / 16);
    passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);

    // Read result
    const resultData = await this.readBuffer(resultBuffer, resultSize * 4);

    // Clean up
    paramsBuffer.destroy();
    aBuffer.destroy();
    bBuffer.destroy();
    resultBuffer.destroy();

    // Convert back to 2D array
    const result: Tensor2d = [];
    for (let i = 0; i < M; i++) {
      result[i] = Array.from(resultData.slice(i * N, (i + 1) * N));
    }

    return result;
  }

  async softmax3D(input: Tensor3d): Promise<Tensor3d> {
    const { findMax, computeExpSum, normalize } = this.softmaxPipelines;
    if (!findMax || !computeExpSum || !normalize) {
      throw new Error('Softmax pipelines not initialized');
    }

    const batchSize = input.length;
    const seqLength = input[0].length;
    const vocabSize = input[0][0].length;
    const totalTokens = batchSize * seqLength;

    // Flatten input
    const inputFlat = new Float32Array(input.flat(2));

    // Create buffers
    const paramsBuffer = this.device.createBuffer({
      size: 16, // 3 * 4 bytes + padding
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([batchSize, seqLength, vocabSize]));

    const inputBuffer = this.createBuffer(inputFlat, GPUBufferUsage.STORAGE);
    const outputBuffer = this.device.createBuffer({
      size: inputFlat.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const maxBuffer = this.device.createBuffer({
      size: totalTokens * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    const sumBuffer = this.device.createBuffer({
      size: totalTokens * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    // Execute three-pass softmax computation
    const commandEncoder = this.device.createCommandEncoder();

    // First pass: find max values (needs params, input, maxValues)
    const bindGroup1 = this.device.createBindGroup({
      layout: findMax.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: inputBuffer } },
        { binding: 3, resource: { buffer: maxBuffer } },
      ],
    });

    let passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(findMax);
    passEncoder.setBindGroup(0, bindGroup1);
    passEncoder.dispatchWorkgroups(Math.ceil(totalTokens / 256));
    passEncoder.end();

    // Second pass: compute exp and sum (needs params, input, output, maxValues, sumValues)
    const bindGroup2 = this.device.createBindGroup({
      layout: computeExpSum.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: inputBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: maxBuffer } },
        { binding: 4, resource: { buffer: sumBuffer } },
      ],
    });

    passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computeExpSum);
    passEncoder.setBindGroup(0, bindGroup2);
    passEncoder.dispatchWorkgroups(Math.ceil(totalTokens / 256));
    passEncoder.end();

    // Third pass: normalize (needs params, output, sumValues)
    const bindGroup3 = this.device.createBindGroup({
      layout: normalize.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: sumBuffer } },
      ],
    });

    passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(normalize);
    passEncoder.setBindGroup(0, bindGroup3);
    passEncoder.dispatchWorkgroups(Math.ceil(inputFlat.length / 256));
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);

    // Read result
    const resultData = await this.readBuffer(outputBuffer, inputFlat.byteLength);

    // Clean up
    paramsBuffer.destroy();
    inputBuffer.destroy();
    outputBuffer.destroy();
    maxBuffer.destroy();
    sumBuffer.destroy();

    // Convert back to 3D array
    const result: Tensor3d = [];
    let idx = 0;
    for (let b = 0; b < batchSize; b++) {
      result[b] = [];
      for (let s = 0; s < seqLength; s++) {
        result[b][s] = Array.from(resultData.slice(idx, idx + vocabSize));
        idx += vocabSize;
      }
    }

    return result;
  }

  async elementwiseAdd(a: Tensor3d, b: Tensor3d): Promise<Tensor3d> {
    if (!this.elementwisePipeline) {
      throw new Error('Element-wise pipeline not initialized');
    }

    // Flatten arrays
    const aFlat = new Float32Array(a.flat(2));
    const bFlat = new Float32Array(b.flat(2));
    const size = aFlat.length;

    // Create buffers
    const paramsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([size, 0])); // operation 0 = add

    const aBuffer = this.createBuffer(aFlat, GPUBufferUsage.STORAGE);
    const bBuffer = this.createBuffer(bFlat, GPUBufferUsage.STORAGE);
    const resultBuffer = this.device.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Execute computation
    const bindGroup = this.device.createBindGroup({
      layout: this.elementwisePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: aBuffer } },
        { binding: 2, resource: { buffer: bBuffer } },
        { binding: 3, resource: { buffer: resultBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.elementwisePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(size / 256));
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);

    // Read result and reconstruct 3D array
    const resultData = await this.readBuffer(resultBuffer, size * 4);

    // Clean up
    paramsBuffer.destroy();
    aBuffer.destroy();
    bBuffer.destroy();
    resultBuffer.destroy();

    // Convert back to 3D array with same shape as input
    const result: Tensor3d = [];
    let idx = 0;
    for (let i = 0; i < a.length; i++) {
      result[i] = [];
      for (let j = 0; j < a[i].length; j++) {
        const length = a[i][j].length;
        result[i][j] = Array.from(resultData.slice(idx, idx + length));
        idx += length;
      }
    }

    return result;
  }

  async layerNorm(input: Tensor3d, gamma: number[], beta: number[], eps = 1e-5): Promise<Tensor3d> {
    const { computeMeans, computeVariances, normalize } = this.layernormPipelines;
    if (!computeMeans || !computeVariances || !normalize) {
      throw new Error('LayerNorm pipelines not initialized');
    }

    const batchSize = input.length;
    const seqLength = input[0].length;
    const embeddingSize = input[0][0].length;
    const totalTokens = batchSize * seqLength;

    // Flatten input
    const inputFlat = new Float32Array(input.flat(2));
    const gammaArray = new Float32Array(gamma);
    const betaArray = new Float32Array(beta);

    // Create buffers
    const paramsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([batchSize, seqLength, embeddingSize]));
    this.device.queue.writeBuffer(paramsBuffer, 12, new Float32Array([eps]));

    const inputBuffer = this.createBuffer(inputFlat, GPUBufferUsage.STORAGE);
    const gammaBuffer = this.createBuffer(gammaArray, GPUBufferUsage.STORAGE);
    const betaBuffer = this.createBuffer(betaArray, GPUBufferUsage.STORAGE);
    const outputBuffer = this.device.createBuffer({
      size: inputFlat.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const meansBuffer = this.device.createBuffer({
      size: totalTokens * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    const variancesBuffer = this.device.createBuffer({
      size: totalTokens * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    // Execute three-pass layer normalization
    const commandEncoder = this.device.createCommandEncoder();

    // First pass: compute means (needs params, input, means)
    const bindGroup1 = this.device.createBindGroup({
      layout: computeMeans.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: inputBuffer } },
        { binding: 5, resource: { buffer: meansBuffer } },
      ],
    });

    let passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computeMeans);
    passEncoder.setBindGroup(0, bindGroup1);
    passEncoder.dispatchWorkgroups(Math.ceil(totalTokens / 256));
    passEncoder.end();

    // Second pass: compute variances (needs params, input, means, variances)
    const bindGroup2 = this.device.createBindGroup({
      layout: computeVariances.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: inputBuffer } },
        { binding: 5, resource: { buffer: meansBuffer } },
        { binding: 6, resource: { buffer: variancesBuffer } },
      ],
    });

    passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computeVariances);
    passEncoder.setBindGroup(0, bindGroup2);
    passEncoder.dispatchWorkgroups(Math.ceil(totalTokens / 256));
    passEncoder.end();

    // Third pass: normalize (needs all buffers)
    const bindGroup3 = this.device.createBindGroup({
      layout: normalize.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: inputBuffer } },
        { binding: 2, resource: { buffer: gammaBuffer } },
        { binding: 3, resource: { buffer: betaBuffer } },
        { binding: 4, resource: { buffer: outputBuffer } },
        { binding: 5, resource: { buffer: meansBuffer } },
        { binding: 6, resource: { buffer: variancesBuffer } },
      ],
    });

    passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(normalize);
    passEncoder.setBindGroup(0, bindGroup3);
    passEncoder.dispatchWorkgroups(Math.ceil(inputFlat.length / 256));
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);

    // Read result
    const resultData = await this.readBuffer(outputBuffer, inputFlat.byteLength);

    // Clean up
    paramsBuffer.destroy();
    inputBuffer.destroy();
    gammaBuffer.destroy();
    betaBuffer.destroy();
    outputBuffer.destroy();
    meansBuffer.destroy();
    variancesBuffer.destroy();

    // Convert back to 3D array
    const result: Tensor3d = [];
    let idx = 0;
    for (let b = 0; b < batchSize; b++) {
      result[b] = [];
      for (let s = 0; s < seqLength; s++) {
        result[b][s] = Array.from(resultData.slice(idx, idx + embeddingSize));
        idx += embeddingSize;
      }
    }

    return result;
  }

  // GPU-accelerated AdamW weight update
  async adamwUpdate(
    weights: Float32Array,
    gradients: Float32Array,
    momentum: Float32Array,
    velocity: Float32Array,
    beta1: number,
    beta2: number,
    eps: number,
    learningRate: number,
    weightDecay: number,
    bc1: number,
    bc2: number,
  ): Promise<void> {
    const size = weights.length;

    if (!this.device || !this.adamwPipeline) {
      await this.initAdamwPipeline(size);
    }

    // Create/update buffers
    const weightsBuffer = this.device!.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    new Float32Array(weightsBuffer.getMappedRange()).set(weights);
    weightsBuffer.unmap();

    const gradientsBuffer = this.device!.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(gradientsBuffer.getMappedRange()).set(gradients);
    gradientsBuffer.unmap();

    const momentumBuffer = this.device!.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    new Float32Array(momentumBuffer.getMappedRange()).set(momentum);
    momentumBuffer.unmap();

    const velocityBuffer = this.device!.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    new Float32Array(velocityBuffer.getMappedRange()).set(velocity);
    velocityBuffer.unmap();

    const paramsBuffer = this.device!.createBuffer({
      size: 8 * 4, // 8 float32s
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(paramsBuffer.getMappedRange()).set([beta1, beta2, eps, learningRate, weightDecay, bc1, bc2, size]);
    paramsBuffer.unmap();

    const bindGroup = this.device!.createBindGroup({
      layout: this.adamwPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: weightsBuffer } },
        { binding: 2, resource: { buffer: gradientsBuffer } },
        { binding: 3, resource: { buffer: momentumBuffer } },
        { binding: 4, resource: { buffer: velocityBuffer } },
      ],
    });

    const commandEncoder = this.device!.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.adamwPipeline!);
    computePass.setBindGroup(0, bindGroup);

    const workgroupSize = 64;
    const numWorkgroups = Math.ceil(size / workgroupSize);
    computePass.dispatchWorkgroups(numWorkgroups);
    computePass.end();

    // Copy results back
    const weightsResultBuffer = this.device!.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const momentumResultBuffer = this.device!.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const velocityResultBuffer = this.device!.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    commandEncoder.copyBufferToBuffer(weightsBuffer, 0, weightsResultBuffer, 0, size * 4);
    commandEncoder.copyBufferToBuffer(momentumBuffer, 0, momentumResultBuffer, 0, size * 4);
    commandEncoder.copyBufferToBuffer(velocityBuffer, 0, velocityResultBuffer, 0, size * 4);

    this.device!.queue.submit([commandEncoder.finish()]);

    // Read results
    await weightsResultBuffer.mapAsync(GPUMapMode.READ);
    await momentumResultBuffer.mapAsync(GPUMapMode.READ);
    await velocityResultBuffer.mapAsync(GPUMapMode.READ);

    weights.set(new Float32Array(weightsResultBuffer.getMappedRange()));
    momentum.set(new Float32Array(momentumResultBuffer.getMappedRange()));
    velocity.set(new Float32Array(velocityResultBuffer.getMappedRange()));

    weightsResultBuffer.unmap();
    momentumResultBuffer.unmap();
    velocityResultBuffer.unmap();
  }

  private adamwPipeline: GPUComputePipeline | null = null;

  private async initAdamwPipeline(_size: number): Promise<void> {
    if (!this.device) {
      throw new Error('GPU device not initialized');
    }

    const adamwShaderCode = `
      struct Params {
        beta1: f32,
        beta2: f32,
        eps: f32,
        learningRate: f32,
        weightDecay: f32,
        bc1: f32,
        bc2: f32,
        size: f32,
      }

      @group(0) @binding(0) var<uniform> params: Params;
      @group(0) @binding(1) var<storage, read_write> weights: array<f32>;
      @group(0) @binding(2) var<storage, read> gradients: array<f32>;
      @group(0) @binding(3) var<storage, read_write> momentum: array<f32>;
      @group(0) @binding(4) var<storage, read_write> velocity: array<f32>;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index = global_id.x;
        let size = u32(params.size);
        
        if (index >= size) {
          return;
        }

        let g = gradients[index];
        
        // Update momentum: m = beta1 * m + (1 - beta1) * g
        momentum[index] = params.beta1 * momentum[index] + (1.0 - params.beta1) * g;
        
        // Update velocity: v = beta2 * v + (1 - beta2) * g^2
        velocity[index] = params.beta2 * velocity[index] + (1.0 - params.beta2) * g * g;
        
        // Bias correction
        let mHat = momentum[index] / params.bc1;
        let vHat = velocity[index] / params.bc2;
        
        // Weight decay: w = w * (1 - lr * decay)
        weights[index] = weights[index] * (1.0 - params.learningRate * params.weightDecay);
        
        // AdamW update: w = w - lr * mHat / (sqrt(vHat) + eps)
        weights[index] = weights[index] - params.learningRate * (mHat / (sqrt(vHat) + params.eps));
      }
    `;

    const shaderModule = this.device.createShaderModule({ code: adamwShaderCode });

    this.adamwPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });
  }
}
