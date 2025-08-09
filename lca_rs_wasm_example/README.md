# lca_rs_wasm_example

Tiny browser demo that loads the WASM builds and runs a sample calculation.

Prereqs:
- Build WASM packages for lca-rs (and deps) first from the workspace roots if needed.

```bash
# from repo root or the crate root
wasm-pack build lca-core -- --features wasm
wasm-pack build lca-rs -- --features wasm
```

Run the example:

```bash
npm install
npm run start
```

Then open the printed URL (defaults to http://localhost:8080). Click “Run LCA Calculation”.

Troubleshooting:
- If the WASM JS files aren’t found, ensure the import path in `index.js` matches `../lca-rs/pkg/lca_rs.js` and the packages exist.
- Check the browser console for any loading errors.
