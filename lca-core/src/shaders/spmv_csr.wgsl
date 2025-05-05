// Sparse Matrix-Vector Multiply (SpMV) for CSR format
// Calculates y = A * x

struct MatrixInfo {
    rows: u32,
    cols: u32,
    nnz: u32,
};

// Using a regular storage buffer for output, as each thread writes to a unique row.
@group(0) @binding(0) var<uniform> matrix_info: MatrixInfo;
// CSR format uses row_pointers instead of row_indices directly per element
@group(0) @binding(1) var<storage, read> row_pointers: array<u32>; // A's row pointers (size rows + 1)
@group(0) @binding(2) var<storage, read> col_indices: array<u32>; // A's col indices (size nnz)
@group(0) @binding(3) var<storage, read> values: array<f64>;      // A's values (size nnz)
@group(0) @binding(4) var<storage, read> x: array<f64>;          // Input vector x (size cols)
@group(0) @binding(5) var<storage, read_write> y_output: array<f64>; // Output vector y (f64, size rows)

// Main SpMV kernel: One thread per row
@compute @workgroup_size(256)
fn main_spmv_csr_per_row(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;

    if (row >= matrix_info.rows) {
        return;
    }

    let row_start = row_pointers[row];
    let row_end = row_pointers[row + 1];
    var sum: f64 = 0.0;

    for (var i = row_start; i < row_end; i = i + 1u) {
        let col = col_indices[i];
        let a_val = values[i];
        let x_val = x[col];
        sum = sum + a_val * x_val;
    }

    // Directly store the final sum for the row into the output buffer.
    // Atomics are not needed here because each thread writes to a unique row index.
    y_output[row] = sum;
}

// Removed initialize_y function as it's no longer needed for non-atomic buffer.
