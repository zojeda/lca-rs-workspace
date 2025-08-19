# lca-rs workspace

GPU-accelerated Life Cycle Assessment (LCA) toolchain in Rust. This monorepo contains core GPU math, linear solvers, domain glue, a web API, data tooling, and runnable examples (native and WASM).

## Projects at a glance

Libraries
- lca-core — Core primitives and GPU compute (wgpu): vectors, sparse matrices, device backends, and LCA domain models.
- lca-lsolver — GPU-accelerated iterative solvers (BiCGSTAB, CG) on top of lca-core.
- lca-rs — Glue crate exposing higher-level LCA helpers on top of lca-core and lca-lsolver; supports native and WASM.

Apps & examples
- lca-webservice — Axum-based API to compile/evaluate LCA models with streaming progress and OpenAPI docs.
- ecoinvent-example — Small example that loads ecoinvent “universal matrix export” CSVs, mixes with a custom system, and evaluates on GPU.
- lca_rs_wasm_example — Minimal browser example that loads the WASM build and runs a calculation.

Data tooling
- ecospold-parser — EcoSpold2 XML parser utilities for import pipelines.
- lca-database — Import and model storage utilities built around SurrealDB, using ecospold-parser.

## How things relate

```mermaid
graph LR
  subgraph Libraries
    core[lca-core]
    lsolver[lca-lsolver]
    glue[lca-rs]
  end
  subgraph Apps & Examples
    web[lca-webservice]
    ei[ecoinvent-example]
    wasm[lca_rs_wasm_example]
  end
  subgraph Data tooling
    parser[ecospold-parser]
    db[lca-database]
  end

  %% Code dependencies (A --> B means A depends on/uses B)
  lsolver --> core
  glue --> core
  glue --> lsolver
  web --> glue
  web --> core
  ei --> glue
  ei --> core
  db --> parser

  %% External data used by examples
  ume[ecoinvent UME CSVs] -. feeds .-> ei
```

Notes
- lca-core re-exports common types for convenience (GpuDevice, SparseMatrix, LcaSystem, etc.).
- All GPU compute goes through wgpu (native, WebGPU in browsers). WASM bindings are gated by features.

## Quick start

Prerequisites
- Rust toolchain (stable). GPU-capable machine recommended for native GPU runs.
- Optional for browser/WASM: Node.js + wasm-pack.
- Optional for ecoinvent example: universal matrix export CSVs (A_public.csv, B_public.csv, C.csv, ee_index.csv, ie_index.csv, LCIA_index.csv).

Build all crates

```bash
cargo build
```

Run the web API

```bash
cargo run -p lca-webservice
# Swagger UI: http://localhost:3000/swagger-ui
# Health:     http://localhost:3000/
```

Run the ecoinvent example

```bash
# Place CSVs under ecoinvent-example/universal_matrix_export/
RUST_LOG=info,wgpu=warn cargo run -p ecoinvent-example
```

WASM (example app)

```bash
# See lca_rs_wasm_example/ for a minimal browser setup using the generated wasm pkg.
```

## Repository layout

- ecoinvent/ — Local datasets and notebooks (e.g., universal matrix export 3.11 cut-off).
- ecoinvent-example/ — Example binary using ecoinvent UME CSVs with GPU evaluation.
- ecospold-parser/ — EcoSpold2 parser library.
- lca-core/ — Core GPU compute, math, devices, and LCA models.
- lca-database/ — Import/storage helpers (SurrealDB) using ecospold-parser.
- lca-lsolver/ — GPU iterative solvers built on lca-core.
- lca-rs/ — Glue and high-level helpers on top of lca-core and lca-lsolver (native + WASM).
- lca-webservice/ — Axum-based REST/SSE service orchestrating model build and evaluation.
- lca_rs_wasm_example/ — Minimal browser example wiring the WASM build.

## Features and targets

- Native (default): full GPU via wgpu; logging via env_logger/tracing.
- WASM: optional features enable wasm-bindgen exports; WebGPU backend enabled in lca-core when targeting wasm32.

## Data: ecoinvent UME

- The example expects the “universal matrix export” CSVs located at `ecoinvent-example/universal_matrix_export/`.
- A separate `ecoinvent/` folder holds larger datasets and analysis artifacts not used directly by the code.

---

For per-crate details, see each project’s README.
