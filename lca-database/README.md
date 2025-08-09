# lca-database

SurrealDB-backed LCA data layer (experimental).

- Provides `SurrealLcaDb` and trait `LcaDatabase` for creating core entities (Database, Geography, Activities, Exchanges, Classifications).
- Types are in `src/model.rs`. Errors in `src/error.rs`.

Run local SurrealDB (example):

```bash
# ensure surrealdb is installed; start script provided
./start_surrealdb.sh
```

Build:

```bash
cargo build -p lca-database
```
