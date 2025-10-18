## lca-core

Core primitives and GPU compute used by solvers and the webservice.

### New module layout

- context: GPU context and buffer helpers (`GpuContext`).
- math: linear algebra building blocks and ops
	- traits: `Matrix`, `Vector`
	- vector: `GpuVector`
	- sparse_matrix: `SparseMatrix`, `SparseMatrixGpu`, `Triplete`
	- ops: internal GPU compute (axpy, dot, spmv, spmv_transpose)
- devices: execution backends
	- `GpuDevice` (+ `TransferStats`), `CpuDevice`, `Device` trait
- models: LCA domain types
	- `LcaMatrix`, `LcaSystem`, `DemandItem`, `InterSystemLink`

Public API re-exports are provided from `lib.rs` so consumers can continue to `use lca_core::{GpuDevice, SparseMatrix, SparseMatrixGpu, GpuVector, LcaMatrix, LcaSystem, Matrix, Vector, LcaCoreError};`.

Compatibility shims: old paths like `lca_core::sparse_matrix::Triplete`, `lca_core::vector::GpuVector`, `lca_core::device::GpuDevice` still work. Please migrate to `lca_core::math::{sparse_matrix::*, vector::*, traits::*}` and `lca_core::devices::*` over time.

### GPU ops

- On `GpuDevice`: `axpy`, `dot`, `elementwise_mul`, `extract_diagonal`.
- On `SparseMatrixGpu`: `spmv`, `spmv_transpose`.

### Build (native)

```bash
cargo build -p lca-core
```

### Build npm package (wasm)

```bash
wasm-pack build -- --features wasm-bindings
```

Example (browser): see `../lca_rs_wasm_example/` for loading and calling into the WASM.