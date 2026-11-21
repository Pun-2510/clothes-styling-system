from pathlib import Path

import pandas as pd
import pytest

from src.metadata_retrieval import MetadataRetriever, build_metadata_index


def _write_catalog(path: Path):
    pd.DataFrame(
        [
            {
                "product_id": "p1",
                "product_name": "Red cotton striped shirt",
                "description": "soft casual button shirt",
                "category": "Shirts",
            },
            {
                "product_id": "p2",
                "product_name": "Black leather shoes",
                "description": "formal lace up footwear",
                "category": "Formal Shoes",
            },
            {
                "product_id": "p3",
                "product_name": "Blue denim shorts",
                "description": "washed casual denim",
                "category": "Shorts",
            },
        ]
    ).to_csv(path, index=False)


def test_metadata_index_scores_catalog_names_and_descriptions(tmp_path):
    catalog = tmp_path / "products.csv"
    index_dir = tmp_path / "metadata_index"
    _write_catalog(catalog)
    metadata = build_metadata_index(catalog, index_dir, min_df=1)

    retriever = MetadataRetriever(index_dir, catalog, expected_rows=3)
    scores, usable = retriever.score("red cotton striped")

    assert metadata["rows"] == 3
    assert retriever.features > 0
    assert usable is True
    assert scores.argmax() == 0


def test_metadata_index_rejects_a_different_catalog(tmp_path):
    catalog = tmp_path / "products.csv"
    index_dir = tmp_path / "metadata_index"
    _write_catalog(catalog)
    build_metadata_index(catalog, index_dir, min_df=1)
    products = pd.read_csv(catalog)
    products.loc[0, "product_name"] = "Changed product"
    products.to_csv(catalog, index=False)

    with pytest.raises(ValueError, match="another catalog"):
        MetadataRetriever(index_dir, catalog, expected_rows=3)


def test_refinement_queries_are_catalog_derived(tmp_path):
    catalog = tmp_path / "products.csv"
    index_dir = tmp_path / "metadata_index"
    _write_catalog(catalog)
    build_metadata_index(catalog, index_dir, min_df=1)
    retriever = MetadataRetriever(index_dir, catalog, expected_rows=3)

    prompts = retriever.refinement_queries_for_rows(
        [0, 1],
        max_terms=2,
        minimum_document_frequency=1,
        maximum_document_ratio=1.0,
    )

    assert len(prompts) == 2
    assert prompts[0]
    assert prompts[1]
