from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

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


class MetadataRetrievalTests(unittest.TestCase):
    def test_metadata_index_scores_catalog_names_and_descriptions(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            catalog = root / "products.csv"
            index_dir = root / "metadata_index"
            _write_catalog(catalog)
            metadata = build_metadata_index(catalog, index_dir, min_df=1)

            retriever = MetadataRetriever(index_dir, catalog, expected_rows=3)
            scores, usable = retriever.score("red cotton striped")

            self.assertEqual(metadata["rows"], 3)
            self.assertGreater(retriever.features, 0)
            self.assertTrue(usable)
            self.assertEqual(int(scores.argmax()), 0)

    def test_metadata_index_rejects_a_different_catalog(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            catalog = root / "products.csv"
            index_dir = root / "metadata_index"
            _write_catalog(catalog)
            build_metadata_index(catalog, index_dir, min_df=1)
            products = pd.read_csv(catalog)
            products.loc[0, "product_name"] = "Changed product"
            products.to_csv(catalog, index=False)

            with self.assertRaisesRegex(ValueError, "another catalog"):
                MetadataRetriever(index_dir, catalog, expected_rows=3)

    def test_refinement_queries_are_catalog_derived(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            catalog = root / "products.csv"
            index_dir = root / "metadata_index"
            _write_catalog(catalog)
            build_metadata_index(catalog, index_dir, min_df=1)
            retriever = MetadataRetriever(index_dir, catalog, expected_rows=3)

            prompts = retriever.refinement_queries_for_rows(
                [0, 1],
                max_terms=2,
                minimum_document_frequency=1,
                maximum_document_ratio=1.0,
            )

            self.assertEqual(len(prompts), 2)
            self.assertTrue(prompts[0])
            self.assertTrue(prompts[1])


if __name__ == "__main__":
    unittest.main()
