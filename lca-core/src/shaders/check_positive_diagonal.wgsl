// Shader to check if all diagonal elements of a sparse matrix are positive.

// Binding groups define how resources (buffers, textures) are accessed.
@group(0) @binding(0) var<storage, read> row_indices: array<u32>; // Input: Row indices of non-zero elements
@group(0) @binding(1) var<storage, read> col_indices: array<u32>; // Input: Column indices of non-zero elements
@group(0) @binding(2) var<storage, read> values: array<f64>;      // Input: Values of non-zero elements
@group(0) @binding(3) var<storage, read_write> result_flag: atomic<u32>; // Output: Flag (0 = all positive, 1 = non-positive found)

// Entry point for the compute shader.
// global_invocation_id.x corresponds to the index of the non-zero element being processed.
@compute @workgroup_size(256) // Adjust workgroup size based on typical GPU capabilities
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let nnz = arrayLength(&values); // Get the total number of non-zero elements
    let index = global_id.x;

    // Ensure we don't go out of bounds
    if (index >= nnz) {
        return;
    }

    let row = row_indices[index];
    let col = col_indices[index];
    let val = values[index];

    // Check if this element is on the diagonal
    if (row == col) {
        // Check if the diagonal value is non-positive
        if (val <= 0.0) {
            // Atomically store 1 in the result flag if a non-positive diagonal is found.
            // Subsequent writes won't change it from 1.
            atomicStore(&result_flag, 1u);
            // Note: We don't need to break or stop other threads,
            // as finding just one is sufficient.
        }
    }
}
