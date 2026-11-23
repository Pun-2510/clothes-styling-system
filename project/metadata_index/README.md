# TF-IDF metadata index

This directory stores the local, row-aligned TF-IDF index generated from
`data/processed/products.csv`. Generated matrices and vectorizer files are
ignored by Git because they must be rebuilt whenever the processed catalog
changes.

Text-only retrieval may use this score across its eligible candidates. In a
combined image-text query, the metadata score is category-gated: only products
in categories supported by image evidence, or one category explicitly stated
in the text, receive the TF-IDF boost. Products outside those categories keep
their raw CLIP fusion score.

Run from `project/`:

```powershell
.\.venv\Scripts\python.exe -m src.build_metadata_index `
  --catalog-csv data/processed/products.csv `
  --output-dir metadata_index `
  --min-df 2 `
  --max-features 50000
```
