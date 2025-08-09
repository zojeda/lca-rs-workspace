# lca-rs

Glue crate exposing LCA model types and evaluation helpers, with optional WASM bindings.

- Public modules: `model`, `error`; re-exports `EvalLCASystem` runner.
- WASM: initializes logging/panic hook via `#[wasm_bindgen(start)]` when built with `--features wasm`.

Build:

```bash
cargo build -p lca-rs
```

WASM (via workspace features from dependents like web or example):
- See `../lca_rs_wasm_example/` for browser usage.
