// Dot Product - Pass 2: Sum partial results from Pass 1

struct Params {
    size: u32, // Number of partial results from Pass 1
};

// Shared memory for reduction
var<workgroup> partial_sums: array<f32, 256>; // Size must match workgroup_size

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> partial_results: array<f32>; // Input: Partial sums from Pass 1
@group(0) @binding(2) var<storage, read_write> final_result: array<f32>; // Output: Single f32 value

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let tid = local_id.x;
    let gid = global_id.x;
    let workgroup_size = 256u;

    // Initialize shared memory
    partial_sums[tid] = 0.0;

    // Load data into shared memory if within bounds
    if (gid < params.size) {
        partial_sums[tid] = partial_results[gid];
    }

    // Synchronize threads within the workgroup
    workgroupBarrier();

    // Perform reduction within the workgroup
    var i = workgroup_size / 2u;
    while (i > 0u) {
        if (tid < i) {
            partial_sums[tid] += partial_sums[tid + i];
        }
        workgroupBarrier();
        i /= 2u;
    }

    // The first thread writes the final sum for this (only) workgroup
    // Assumes Pass 2 runs with only ONE workgroup.
    if (tid == 0u) {
        final_result[0] = partial_sums[0];
    }
}
