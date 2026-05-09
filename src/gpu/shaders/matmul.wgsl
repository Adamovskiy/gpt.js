// Matrix multiplication shader for GPU acceleration
struct Params {
  M: u32,  // rows of A
  N: u32,  // cols of B
  K: u32,  // cols of A / rows of B
};

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> matrixA: array<f32>;

@group(0) @binding(2)
var<storage, read> matrixB: array<f32>;

@group(0) @binding(3)
var<storage, read_write> result: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let row = globalId.x;
  let col = globalId.y;
  
  if (row >= params.M || col >= params.N) {
    return;
  }
  
  var sum = 0.0;
  for (var k = 0u; k < params.K; k++) {
    let aIndex = row * params.K + k;
    let bIndex = k * params.N + col;
    sum += matrixA[aIndex] * matrixB[bIndex];
  }
  
  let resultIndex = row * params.N + col;
  result[resultIndex] = sum;
}