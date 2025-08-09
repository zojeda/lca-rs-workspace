# ecoinvent-example

Example binary combining a small custom system with ecoinvent matrices (universal matrix export) to run a GPU evaluation.

Prereqs:
- Place the universal_matrix_export files under `./universal_matrix_export/` (A_public.csv, B_public.csv, C.csv, ee_index.csv, ie_index.csv, LCIA_index.csv).

Run:

```bash
RUST_LOG=info,wgpu=warn cargo run -p ecoinvent-example
```

Notes:
- This example constructs an LcaSystem from CSV matrices, merges with a small in-memory system, and evaluates with a GpuDevice.
