// Element-wise operations shader
struct Params {
  size: u32,
  operation: u32,  // 0 = add, 1 = multiply, 2 = relu
};

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> inputA: array<f32>;

@group(0) @binding(2)
var<storage, read> inputB: array<f32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let idx = globalId.x;
  
  if (idx >= params.size) {
    return;
  }
  
  let a = inputA[idx];
  let b = inputB[idx];
  
  switch (params.operation) {
    case 0u: { // Add
      output[idx] = a + b;
    }
    case 1u: { // Multiply
      output[idx] = a * b;
    }
    case 2u: { // ReLU on inputA
      output[idx] = max(0.0, a);
    }
    default: {
      output[idx] = a;
    }
  }
}