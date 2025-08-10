# lca-rs

Glue crate exposing LCA model helpers atop `lca-core` (which now contains the domain models).

- LCA domain types live in `lca-core::models` and are re-exported from `lca-core` root for convenience (`LcaMatrix`, `LcaSystem`, `DemandItem`, `InterSystemLink`).
- WASM: initializes logging/panic hook via `#[wasm_bindgen(start)]` when built with `--features wasm`.

Build:

```bash
cargo build -p lca-rs
```

WASM (via workspace features from dependents like web or example):
- See `../lca_rs_wasm_example/` for browser usage.
