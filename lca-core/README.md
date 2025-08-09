## lca-core

Core GPU primitives and operations used by solvers and the webservice. Provides:
- GpuDevice async constructor and GPU ops (axpy, dot, elementwise_mul, extract_diagonal)
- SparseMatrix (CPU) and SparseMatrixGpu (GPU) with `spmv`/`spmv_transpose`
- GpuVector with read/write/clone helpers

Targets/features:
- native (default): Vulkan/Metal/DX12/WebGPU via wgpu
- wasm: wasm-bindgen exports for browser

### Build (native)

```bash
cargo build -p lca-core
```

### Build npm package (wasm)

```bash
wasm-pack build -- --features wasm
```

Example (browser): see `../lca_rs_wasm_example/` for loading and calling into the WASM.