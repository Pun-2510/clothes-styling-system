"""Small offline fixtures for deterministic catalog preparation."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from PIL import Image

from src.clip_data import file_sha256
from src.prepare_dataset import prepare_catalog, sample_products


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        rows = []
        for index in range(12):
            path = self.root / f"{index}.png"
            Image.new("RGB", (8, 8), (index * 19, 20, 80)).save(path)
            rows.append(
                {
                    "product_id": f"p{index}",
                    "product_name": f"bag {index}",
                    "category": "bags" if index < 8 else "shoes",
                    "image_path": str(path),
                    "image_sha256": file_sha256(path),
                }
            )
        self.catalog = pd.DataFrame(rows)
        self.csv = self.root / "raw.csv"
        self.catalog.to_csv(self.csv, index=False)

    def test_exact_sampling_repeatable_and_preserves_categories(self):
        for size in range(1, 13):
            first = sample_products(self.catalog, size)
            self.assertEqual(len(first), size)
            self.assertEqual(first.product_id.nunique(), size)
            pd.testing.assert_frame_equal(first, sample_products(self.catalog, size))
            if size >= 2:
                self.assertEqual(first.category.nunique(), 2)
        with self.assertRaises(ValueError):
            sample_products(self.catalog, 13)
        self.assertEqual(len(sample_products(self.catalog, None)), 12)

    def test_prepare_exact_after_invalid_and_duplicate_images(self):
        duplicate = self.catalog.iloc[0].copy()
        duplicate_path = self.root / "duplicate.png"
        duplicate_path.write_bytes((self.root / "0.png").read_bytes())
        duplicate["image_path"] = str(duplicate_path)
        broken = duplicate.copy()
        broken_path = self.root / "broken.png"
        broken_path.write_bytes(b"not an image")
        broken["image_path"] = str(broken_path)
        data = pd.concat(
            [self.catalog, pd.DataFrame([duplicate, broken])],
            ignore_index=True,
        )
        data.to_csv(self.csv, index=False)

        destination = self.root / "prepared" / "products.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            products = prepare_catalog(
                self.csv,
                self.root,
                destination,
                6,
                categories=["bags"],
            )
        self.assertEqual(len(products), 6)
        self.assertEqual(products.category.unique().tolist(), ["bags"])
        self.assertEqual(products.image_sha256.nunique(), 6)

        summary = json.loads(
            destination.with_suffix(".preparation.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["eligible_products"], 8)
        self.assertEqual(summary["invalid_images"], 1)
        self.assertEqual(summary["duplicate_images_removed"], 1)

        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(
            ValueError
        ):
            prepare_catalog(
                self.csv,
                self.root,
                self.root / "too_many.csv",
                9,
                categories=["bags"],
            )
        self.assertFalse((self.root / "too_many.csv").exists())


if __name__ == "__main__":
    unittest.main()
