// Dot Product - Pass 1: Reduce within workgroups
// Calculates dot(x, y)

struct Params {
    size: u32,
};

// Shared memory within the workgroup for intermediate sums
var<workgroup> partial_sums: array<f64, 256>; // Size must match workgroup_size

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f64>;
@group(0) @binding(2) var<storage, read> y: array<f64>;
@group(0) @binding(3) var<storage, read_write> partial_results: array<f64>; // Output buffer for partial sums

@compute @workgroup_size(256) // workgroup_size_x
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let tid = local_id.x; // Thread ID within the workgroup
    let gid = global_id.x; // Global thread ID
    let group_idx = group_id.x; // Workgroup ID
    let workgroup_size = 256u; // Must match @workgroup_size

    // Initialize shared memory for this thread
    partial_sums[tid] = 0.0;

    // Each thread calculates its contribution if within bounds
    if (gid < params.size) {
        partial_sums[tid] = x[gid] * y[gid];
    }

    // Synchronize threads within the workgroup to ensure all reads are done
    workgroupBarrier();

    // Perform reduction within the workgroup
    // Reduce sums in shared memory using a tree reduction
    var i = workgroup_size / 2u;
    while (i > 0u) {
        if (tid < i) {
            partial_sums[tid] += partial_sums[tid + i];
        }
        workgroupBarrier(); // Sync after each reduction step
        i /= 2u;
    }

    // The first thread in the workgroup writes the final sum for this group
    // to the output buffer
    if (tid == 0u) {
        partial_results[group_idx] = partial_sums[0];
    }
}
