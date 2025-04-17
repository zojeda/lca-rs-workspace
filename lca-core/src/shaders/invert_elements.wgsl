// WGSL Shader to compute element-wise inverse (1.0 / x)

@group(0) @binding(0) var<storage, read> input_vector: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_vector: array<f32>;

// Define a small epsilon for zero check to avoid division by zero issues
// and handle cases where the value is extremely small.
// Explicitly define as f32 using a suffix or type declaration if supported,
// or ensure the literal implies f32 if suffixes aren't standard WGSL yet.
// Revert to const (defaults to f32)
const EPSILON = 1.0e-15; // Adjust as needed

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    // Bounds check
    let n = arrayLength(&input_vector);
    if (index >= n) {
        return;
    }

    let input_value = input_vector[index];

    // Check if the input value is close to zero by comparing with EPSILON cast to f32
    if (abs(input_value) < f32(EPSILON)) {
        // Handle zero or near-zero case: output 0.0 or another suitable value.
        // Outputting 0.0 is often safe for Jacobi preconditioning, as it effectively
        // skips preconditioning for that row/column.
        // Ensure 0.0 is treated as f32
        output_vector[index] = f32(0.0);
    } else {
        // Ensure 1.0 is treated as f32
        output_vector[index] = f32(1.0) / input_value;
    }
}
