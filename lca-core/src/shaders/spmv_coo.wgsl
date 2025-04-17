// Sparse Matrix-Vector Multiply (SpMV) for COO format
// Calculates y = A * x

struct MatrixInfo {
    rows: u32,
    cols: u32, // Usually same as rows for CG
    nnz: u32,
};

// Using atomic f32 requires specific features/extensions.
// We'll use atomic u32 and reinterpret bits for f32 atomics.
// This is a common workaround.

@group(0) @binding(0) var<uniform> matrix_info: MatrixInfo;
@group(0) @binding(1) var<storage, read> row_indices: array<u32>; // A's row indices
@group(0) @binding(2) var<storage, read> col_indices: array<u32>; // A's col indices
@group(0) @binding(3) var<storage, read> values: array<f32>;      // A's values
@group(0) @binding(4) var<storage, read> x: array<f32>;          // Input vector x
@group(0) @binding(5) var<storage, read_write> y_atomic: array<atomic<u32>>; // Output vector y (atomic u32)


@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x; // Index of the non-zero element

    if (index >= matrix_info.nnz) {
        return;
    }

    let row = row_indices[index];
    let col = col_indices[index];
    let a_val = values[index];
    let x_val = x[col]; // Get corresponding element from x

    // Atomically add the contribution (a_val * x_val) to the correct row in y
    // y[row] += a_val * x_val;
    // Inlined atomicAddf32 logic:
    let value_to_add = a_val * x_val;
    var current_val_u32: u32;
    var new_val_u32: u32;
    loop {
        current_val_u32 = atomicLoad(&y_atomic[row]);
        let current_val_f32 = bitcast<f32>(current_val_u32);
        let new_val_f32 = current_val_f32 + value_to_add;
        new_val_u32 = bitcast<u32>(new_val_f32);

        let compare_result = atomicCompareExchangeWeak(&y_atomic[row], current_val_u32, new_val_u32);
        if (compare_result.exchanged) {
            break;
        }
    }
}

// --- Initialization Shader ---
// We need a separate kernel to initialize the atomic y buffer to zero bits.
@compute @workgroup_size(256)
fn initialize_y(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= matrix_info.rows) {
        return;
    }
    // Initialize with the bit pattern for 0.0f32
    atomicStore(&y_atomic[index], 0u);
}
