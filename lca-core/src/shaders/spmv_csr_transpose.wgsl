// Computes y = A^T * x, where A is a sparse matrix in CSR format.
// y_out is the output vector (length params.cols, i.e., columns of A)
// x_in is the input vector (length params.rows, i.e., rows of A)

// Parameters for the original matrix A
struct SpmvParams {
    rows: u32, // Number of rows in original matrix A
    cols: u32, // Number of columns in original matrix A
    nnz: u32,  // Number of non-zero elements in A
    _padding: u32, // Ensure struct size is multiple of 16 bytes if necessary, matches Rust
};

@group(0) @binding(0) var<uniform> params: SpmvParams;
@group(0) @binding(1) var<storage, read> row_pointers: array<u32>; // CSR row pointers for A (length params.rows + 1)
@group(0) @binding(2) var<storage, read> col_indices: array<u32>;  // CSR col indices for A (length params.nnz)
@group(0) @binding(3) var<storage, read> values: array<f64>;       // CSR values for A (length params.nnz)
@group(0) @binding(4) var<storage, read> x_in: array<f64>;         // Input vector x (length params.rows)
@group(0) @binding(5) var<storage, read_write> y_out: array<f64>;  // Output vector y (length params.cols)

@compute @workgroup_size(256, 1, 1) // Workgroup size can be tuned
fn main_spmv_csr_transpose_per_col(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = global_id.x; // This is the target index for y_out, corresponding to a column in A.
                         // y_out[j] = sum over i (A_ij * x_in[i])

    if (j >= params.cols) { // Ensure j is within the bounds of y_out (params.cols)
        return;
    }

    var sum_val: f64 = 0.0;

    // Iterate over each row 'i' of the original matrix A.
    // For the current output element y_out[j], we need to find all A_ij elements.
    for (var i: u32 = 0u; i < params.rows; i = i + 1u) {
        let row_start_ptr = row_pointers[i];
        let row_end_ptr = row_pointers[i+1u];

        // Scan through the non-zero elements of row 'i' of matrix A.
        // If an element A_ik has k == j, then A_ij is found.
        for (var k_ptr: u32 = row_start_ptr; k_ptr < row_end_ptr; k_ptr = k_ptr + 1u) {
            if (col_indices[k_ptr] == j) {
                sum_val = sum_val + values[k_ptr] * x_in[i];
                // Assuming column indices within a CSR row are sorted.
                // If so, once we find col_indices[k_ptr] == j, this is the only A_ij in this row.
                break; 
            }
            // Optional optimization: if col_indices are sorted and col_indices[k_ptr] > j,
            // we can break early from this inner loop for row 'i' as no further A_ij will be found.
            // if (col_indices[k_ptr] > j) {
            //     break;
            // }
        }
    }
    y_out[j] = sum_val;
}
