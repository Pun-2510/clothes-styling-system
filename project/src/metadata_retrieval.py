"""Catalog-derived TF-IDF metadata refinement for text-bearing queries.

The index is an offline artifact.  Its row order must match ``products.csv``
because every returned score is aligned with the CLIP image/text embeddings.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.clip_data import file_sha256


INDEX_VERSION = 1
VECTORIZER_FILE = "tfidf_vectorizer.joblib"
MATRIX_FILE = "catalog_tfidf.npz"
METADATA_FILE = "metadata.json"


def catalog_metadata_text(products: pd.DataFrame) -> pd.Series:
    """Return the catalog text used by lexical refinement.

    Category is deliberately excluded.  Explicit category phrases are handled
    by the category policy, so TF-IDF focuses on product names and descriptions
    rather than counting the same category evidence twice.
    """
    columns = [
        column
        for column in ("product_name", "description")
        if column in products.columns
    ]
    if not columns:
        raise ValueError(
            "Catalog must contain product_name or description for TF-IDF indexing."
        )
    return (
        products[columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def build_metadata_index(
    catalog_csv: Path,
    output_dir: Path,
    *,
    min_df: int = 2,
    max_df: float = 0.98,
    max_features: int | None = 50_000,
) -> dict:
    """Fit and persist a row-aligned TF-IDF catalog index."""
    catalog_csv = Path(catalog_csv).resolve()
    output_dir = Path(output_dir).resolve()
    products = pd.read_csv(catalog_csv)
    if products.empty:
        raise ValueError("Cannot build a metadata index from an empty catalog.")

    documents = catalog_metadata_text(products)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=int(min_df),
        max_df=float(max_df),
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9-]+\b",
    )
    matrix = vectorizer.fit_transform(documents).tocsr().astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output_dir / VECTORIZER_FILE)
    sparse.save_npz(output_dir / MATRIX_FILE, matrix, compressed=True)

    metadata = {
        "version": INDEX_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_csv": str(catalog_csv),
        "catalog_sha256": file_sha256(catalog_csv),
        "rows": int(matrix.shape[0]),
        "features": int(matrix.shape[1]),
        "text_columns": [
            column
            for column in ("product_name", "description")
            if column in products.columns
        ],
        "parameters": {
            "ngram_range": [1, 2],
            "min_df": int(min_df),
            "max_df": float(max_df),
            "max_features": max_features,
            "sublinear_tf": True,
            "norm": "l2",
        },
    }
    (output_dir / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


class MetadataRetriever:
    """Load, validate and query one persisted TF-IDF catalog index."""

    def __init__(self, index_dir: Path, catalog_csv: Path, expected_rows: int):
        self.index_dir = Path(index_dir).resolve()
        self.catalog_csv = Path(catalog_csv).resolve()
        metadata_path = self.index_dir / METADATA_FILE
        if not metadata_path.is_file():
            raise FileNotFoundError(f"TF-IDF metadata is missing: {metadata_path}")

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("version") != INDEX_VERSION:
            raise ValueError("Unsupported TF-IDF metadata index version.")
        if int(self.metadata.get("rows", -1)) != int(expected_rows):
            raise ValueError("TF-IDF row count does not match products.csv.")
        if self.metadata.get("catalog_sha256") != file_sha256(self.catalog_csv):
            raise ValueError(
                "TF-IDF index was built from another catalog; rebuild metadata_index."
            )

        self.vectorizer = joblib.load(self.index_dir / VECTORIZER_FILE)
        self.matrix = sparse.load_npz(self.index_dir / MATRIX_FILE).tocsr()
        if self.matrix.shape[0] != int(expected_rows):
            raise ValueError("TF-IDF matrix row order is incompatible with the catalog.")

    @property
    def features(self) -> int:
        return int(self.matrix.shape[1])

    def score(self, query: str) -> tuple[np.ndarray, bool]:
        """Return cosine-equivalent TF-IDF scores and whether terms were found."""
        query_vector = self.vectorizer.transform([str(query).strip()])
        if query_vector.nnz == 0:
            return np.zeros(self.matrix.shape[0], dtype=np.float32), False
        scores = (query_vector @ self.matrix.T).toarray()[0]
        return np.asarray(scores, dtype=np.float32), True

    def score_many(self, queries: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized form used by validation-weight tuning."""
        query_matrix = self.vectorizer.transform([str(value) for value in queries])
        usable = np.asarray(query_matrix.getnnz(axis=1) > 0, dtype=bool)
        scores = (query_matrix @ self.matrix.T).toarray().astype(np.float32)
        return scores, usable

    def refinement_queries_for_rows(
        self,
        row_indices,
        *,
        max_terms: int = 3,
        minimum_document_frequency: int | None = None,
        maximum_document_ratio: float = 0.30,
    ) -> list[str]:
        """Derive short validation prompts from the fitted catalog vocabulary.

        This replaces a hand-written colour/pattern dictionary in weight
        selection.  Terms must occur in several catalog rows so one-off product
        identifiers do not become unrealistic combined-query prompts.
        """
        row_indices = np.asarray(row_indices, dtype=np.int64)
        feature_names = np.asarray(self.vectorizer.get_feature_names_out())
        document_frequency = np.asarray(
            self.matrix.getnnz(axis=0), dtype=np.int64
        ).reshape(-1)
        row_count = int(self.matrix.shape[0])
        minimum_df = (
            int(minimum_document_frequency)
            if minimum_document_frequency is not None
            else max(3, int(round(row_count * 0.001)))
        )
        maximum_df = max(minimum_df, int(row_count * maximum_document_ratio))

        prompts = []
        for row_index in row_indices:
            row = self.matrix.getrow(int(row_index))
            ranked = row.indices[np.argsort(-row.data, kind="stable")]
            selected = []
            selected_words = set()
            for feature_index in ranked:
                if not minimum_df <= document_frequency[feature_index] <= maximum_df:
                    continue
                feature = str(feature_names[feature_index]).strip()
                words = set(feature.split())
                if not feature or words <= selected_words:
                    continue
                selected.append(feature)
                selected_words.update(words)
                if len(selected) >= int(max_terms):
                    break
            prompts.append(" ".join(selected))
        return prompts
