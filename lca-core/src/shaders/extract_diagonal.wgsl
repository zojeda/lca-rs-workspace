// WGSL Shader to extract the diagonal elements from a CSR sparse matrix

// Bindings for CSR matrix data
@group(0) @binding(0) var<storage, read> row_pointers: array<u32>; // CSR row pointers (size N+1)
@group(0) @binding(1) var<storage, read> col_indices: array<u32>; // CSR column indices (size NNZ)
@group(0) @binding(2) var<storage, read> values: array<f32>;      // CSR values (size NNZ)

// Binding for the output diagonal vector
@group(0) @binding(3) var<storage, read_write> diagonal_output: array<f32>; // Output vector (size N)

// We could pass N via uniforms, but reading array_length is often simpler if possible,
// or assume the dispatch size matches N. Let's assume dispatch size matches N.

@compute @workgroup_size(64) // Example workgroup size, can be tuned
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_index = global_id.x;

    // Ensure we don't go out of bounds for the output vector / row pointers
    // This assumes dispatch size is exactly N (number of rows/diagonal elements)
    let num_rows = arrayLength(&row_pointers) - 1u; // Get N from row_pointers length
    if (row_index >= num_rows) {
        return;
    }

    let row_start = row_pointers[row_index];
    let row_end = row_pointers[row_index + 1u];

    var found_diagonal = false;
    // Explicitly declare diagonal_value as f32
    var diagonal_value: f32 = 0.0;

    // Iterate through the non-zero elements of the current row
    for (var j = row_start; j < row_end; j = j + 1u) {
        if (col_indices[j] == row_index) {
            diagonal_value = values[j];
            found_diagonal = true;
            break; // Found the diagonal element, no need to search further
        }
    }

    // Write the found diagonal value (or 0.0 if not found) to the output vector
    // No explicit cast needed now as types should match
    diagonal_output[row_index] = diagonal_value;
}
