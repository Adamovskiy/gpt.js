import shaderCode from '@/gpu/shaders/sum.wgsl?raw';

let devicePromise: Promise<GPUDevice> | null = null;

function getDevice(): Promise<GPUDevice> {
  if (!devicePromise) {
    devicePromise = (async () => {
      if (!('gpu' in navigator)) {
        throw new Error('WebGPU is not available in this browser/context.');
      }

      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        throw new Error('No suitable GPU adapter found.');
      }

      return adapter.requestDevice();
    })();
  }

  return devicePromise;
}

async function sum1ToIOnGpu(i: number): Promise<number> {
  const device = await getDevice();

  const count = i;
  const bufferSize = Math.max(4, count * 4);

  const paramsArray = new Uint32Array([i]);

  const paramsBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(paramsBuffer, 0, paramsArray);

  const partialSumsBuffer = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const readbackBuffer = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const shaderModule = device.createShaderModule({
    code: shaderCode,
  });

  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shaderModule,
      entryPoint: 'main',
    },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: paramsBuffer,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: partialSumsBuffer,
        },
      },
    ],
  });

  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);

  const workgroupSize = 256;
  const workgroupCount = Math.ceil(count / workgroupSize);

  if (workgroupCount > 0) {
    passEncoder.dispatchWorkgroups(workgroupCount);
  }

  passEncoder.end();

  commandEncoder.copyBufferToBuffer(partialSumsBuffer, 0, readbackBuffer, 0, bufferSize);

  device.queue.submit([commandEncoder.finish()]);

  await readbackBuffer.mapAsync(GPUMapMode.READ);

  const mappedRange = readbackBuffer.getMappedRange();
  const values = new Uint32Array(mappedRange.slice(0));

  readbackBuffer.unmap();

  let sum = 0;
  for (let n = 0; n < count; n++) {
    sum += values[n];
  }

  paramsBuffer.destroy();
  partialSumsBuffer.destroy();
  readbackBuffer.destroy();

  return sum;
}

self.onmessage = () => {
  // An example of how to call WGSL
  sum1ToIOnGpu(10_000).then((result) => {
    console.log('Received message', result);
  });

  for (let i = 0; i < 10_000; i++) {
    const value = heavyCalculation(i);

    // An example of how to communicate with a web worker
    self.postMessage({
      index: i,
      value,
    });
  }
};

function heavyCalculation(i: number) {
  return Math.sin(i);
}
