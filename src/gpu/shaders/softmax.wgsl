// Softmax computation shader
struct Params {
  batchSize: u32,
  seqLength: u32,
  vocabSize: u32,
};

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> input: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<storage, read_write> maxValues: array<f32>;

@group(0) @binding(4)
var<storage, read_write> sumValues: array<f32>;

// First pass: find maximum values
@compute @workgroup_size(256)
fn findMax(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let batchSeqIdx = globalId.x;
  
  if (batchSeqIdx >= params.batchSize * params.seqLength) {
    return;
  }
  
  let startIdx = batchSeqIdx * params.vocabSize;
  var maxVal = input[startIdx];
  
  for (var i = 1u; i < params.vocabSize; i++) {
    maxVal = max(maxVal, input[startIdx + i]);
  }
  
  maxValues[batchSeqIdx] = maxVal;
}

// Second pass: compute exp(x - max) and sum
@compute @workgroup_size(256)
fn computeExpSum(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let batchSeqIdx = globalId.x;
  
  if (batchSeqIdx >= params.batchSize * params.seqLength) {
    return;
  }
  
  let startIdx = batchSeqIdx * params.vocabSize;
  let maxVal = maxValues[batchSeqIdx];
  var sum = 0.0;
  
  for (var i = 0u; i < params.vocabSize; i++) {
    let expVal = exp(input[startIdx + i] - maxVal);
    output[startIdx + i] = expVal;
    sum += expVal;
  }
  
  sumValues[batchSeqIdx] = sum;
}

// Third pass: normalize by sum
@compute @workgroup_size(256)
fn normalize(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let idx = globalId.x;
  
  if (idx >= params.batchSize * params.seqLength * params.vocabSize) {
    return;
  }
  
  let batchSeqIdx = idx / params.vocabSize;
  let sum = sumValues[batchSeqIdx];
  
  output[idx] = output[idx] / sum;
}