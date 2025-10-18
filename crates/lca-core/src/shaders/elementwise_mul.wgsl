// WGSL Shader to compute element-wise multiplication z = x * y

@group(0) @binding(0) var<storage, read> x: array<f64>;
@group(0) @binding(1) var<storage, read> y: array<f64>;
@group(0) @binding(2) var<storage, read_write> z: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    // Bounds check (assuming all vectors have the same length)
    let n = arrayLength(&x);
    if (index >= n) {
        return;
    }

    z[index] = x[index] * y[index];
}
