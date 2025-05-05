// Performs the AXPY operation: y = alpha * x + y

struct Params {
    alpha: f64,
    size: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f64>;
@group(0) @binding(2) var<storage, read_write> y: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if (index >= params.size) {
        return;
    }

    y[index] = params.alpha * x[index] + y[index];
}
