// WGSL usage demo
struct Params {
  i: u32,
};

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read_write> partialSums: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x + 1u;

  if (index > params.i) {
    return;
  }

  partialSums[globalId.x] = index;
}