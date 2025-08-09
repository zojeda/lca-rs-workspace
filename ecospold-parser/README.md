# ecospold-parser

EcoSpold2 XML parser utilities used for database import.

- `parse_ecospold(&str) -> EcoSpold`
- `parse_master<T>(&str) -> T` for master data files

Build:

```bash
cargo build -p ecospold-parser
```

Notes:
- Handles optional UTF-8 BOM at the start of XML documents.
