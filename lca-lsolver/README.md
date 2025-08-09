# lca-lsolver

GPU-accelerated linear solvers for sparse systems, built on top of lca-core (wgpu).

Status: active WIP. GPU BiCGSTAB and CG are available for `SparseMatrixGpu<f64>`. The binary at `src/main.rs` is a placeholder (prints "Hello, world!").

## What’s here

- Algorithms (async):
  - BiCGSTAB with optional Jacobi preconditioner
  - Conjugate Gradient (assumes SPD)
- Matrix types: use `lca_lsolver::SparseMatrix` on CPU and upload to GPU with `GpuDevice::create_sparse_matrix` to get `SparseMatrixGpu`.
- Device ops from lca-core: axpy, dot, elementwise_mul, extract_diagonal, spmv on SparseMatrixGpu.

## Quick start (BiCGSTAB on GPU)

```rust
use lca_lsolver::{
    algorithms::{BiCGSTAB, SolveAlgorithm},
    GpuDevice, SparseMatrix,
};
use lca_core::sparse_matrix::Triplete;

fn make_tridiagonal(n: usize) -> SparseMatrix {
    let mut t = Vec::new();
    for i in 0..n {
        if i > 0 { t.push(Triplete::new(i, i-1, -1.0)); }
        t.push(Triplete::new(i, i, 2.0));
        if i + 1 < n { t.push(Triplete::new(i, i+1, -1.0)); }
    }
    SparseMatrix::from_triplets(n, n, t).unwrap()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1000;
    let a_cpu = make_tridiagonal(n);
    let b = vec![1.0; n];

    let device = GpuDevice::new().await?;
    let a_gpu = device.create_sparse_matrix(&a_cpu)?;

    let algo = BiCGSTAB::with_params(1e-6, 2*n, false);
    let out = algo.solve(&device, &a_gpu, &b).await?;
    println!("iters={}, residual={:.3e}", out.metadata.iterations, out.metadata.residual_norm);
    Ok(())
}
```

## Run the included example

```bash
# optional: control logs
RUST_LOG=info,wgpu=warn \
cargo run -p lca-lsolver --example pentadiagonal_solve
```

## Features and targets

- Default feature `native`; `wasm` feature enables wasm-bindgen bindings.
- Uses `f64` throughout for matrix/vector data.

## Notes

- Construct CPU matrices with `SparseMatrix::from_triplets` or `from_csr`, then upload via `GpuDevice::create_sparse_matrix`.
- CG assumes SPD; `gpu_sparse_cg_checked::check_symmetry_and_positive_diagonal` is currently a no-op placeholder.
- The crate re-exports core types from lca-core: `GpuDevice`, `SparseMatrix`, `SparseMatrixGpu`, etc.

