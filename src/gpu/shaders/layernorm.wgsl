// Layer normalization shader
struct Params {
  batchSize: u32,
  seqLength: u32,
  embeddingSize: u32,
  eps: f32,
};

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> input: array<f32>;

@group(0) @binding(2)
var<storage, read> gamma: array<f32>;

@group(0) @binding(3)
var<storage, read> beta: array<f32>;

@group(0) @binding(4)
var<storage, read_write> output: array<f32>;

@group(0) @binding(5)
var<storage, read_write> means: array<f32>;

@group(0) @binding(6)
var<storage, read_write> variances: array<f32>;

// First pass: compute means
@compute @workgroup_size(256)
fn computeMeans(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let tokenIdx = globalId.x;
  let totalTokens = params.batchSize * params.seqLength;
  
  if (tokenIdx >= totalTokens) {
    return;
  }
  
  let startIdx = tokenIdx * params.embeddingSize;
  var sum = 0.0;
  
  for (var i = 0u; i < params.embeddingSize; i++) {
    sum += input[startIdx + i];
  }
  
  means[tokenIdx] = sum / f32(params.embeddingSize);
}

// Second pass: compute variances
@compute @workgroup_size(256)
fn computeVariances(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let tokenIdx = globalId.x;
  let totalTokens = params.batchSize * params.seqLength;
  
  if (tokenIdx >= totalTokens) {
    return;
  }
  
  let startIdx = tokenIdx * params.embeddingSize;
  let mean = means[tokenIdx];
  var sumSquares = 0.0;
  
  for (var i = 0u; i < params.embeddingSize; i++) {
    let diff = input[startIdx + i] - mean;
    sumSquares += diff * diff;
  }
  
  variances[tokenIdx] = sumSquares / f32(params.embeddingSize);
}

// Third pass: normalize
@compute @workgroup_size(256)
fn normalize(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let idx = globalId.x;
  let totalElements = params.batchSize * params.seqLength * params.embeddingSize;
  
  if (idx >= totalElements) {
    return;
  }
  
  let tokenIdx = idx / params.embeddingSize;
  let dimIdx = idx % params.embeddingSize;
  
  let mean = means[tokenIdx];
  let variance = variances[tokenIdx];
  let stdDev = sqrt(variance + params.eps);
  
  let normalized = (input[idx] - mean) / stdDev;
  output[idx] = normalized * gamma[dimIdx] + beta[dimIdx];
}