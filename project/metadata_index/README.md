# TF-IDF metadata index

This directory stores the local, row-aligned TF-IDF index generated from
`data/processed/products.csv`. Generated matrices and vectorizer files are
ignored by Git because they must be rebuilt whenever the processed catalog
changes.

Run from `project/`:

```powershell
.\.venv\Scripts\python.exe -m src.build_metadata_index `
  --catalog-csv data/processed/products.csv `
  --output-dir metadata_index `
  --min-df 2 `
  --max-features 50000
```
