"""Small offline fixtures only: no training, downloads, or real catalog encoding."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from comparisons.compare_clip import catalog_test_rows, load_cached_pair, main as compare_main
from comparisons.run_pipeline import build_commands, main as pipeline_main, parse_args, website_environment
from src.clip_data import file_sha256
from src.clip_runtime import write_embedding_metadata
from src.prepare_dataset import prepare_catalog, sample_products


class ComparisonWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        rows = []
        for index in range(12):
            path = self.root / f"{index}.png"
            Image.new("RGB", (8, 8), (index * 19, 20, 80)).save(path)
            rows.append({"product_id": f"p{index}", "product_name": f"bag {index}",
                         "category": "bags" if index < 8 else "shoes", "image_path": str(path),
                         "image_sha256": file_sha256(path)})
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
        data = pd.concat([self.catalog, pd.DataFrame([duplicate, broken])], ignore_index=True)
        data.to_csv(self.csv, index=False)
        destination = self.root / "prepared" / "products.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            products = prepare_catalog(self.csv, self.root, destination, 6, categories=["bags"])
        self.assertEqual(len(products), 6)
        self.assertEqual(products.category.unique().tolist(), ["bags"])
        self.assertEqual(products.image_sha256.nunique(), 6)
        summary = json.loads(destination.with_suffix(".preparation.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["eligible_products"], 8)
        self.assertEqual(summary["invalid_images"], 1)
        self.assertEqual(summary["duplicate_images_removed"], 1)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            prepare_catalog(self.csv, self.root, self.root / "too_many.csv", 9, categories=["bags"])
        self.assertFalse((self.root / "too_many.csv").exists())

    def test_original_test_intersection_and_cache_order(self):
        test = self.catalog.iloc[[1, 3, 5]].copy()
        catalog = self.catalog.iloc[[3, 0, 1]].reset_index(drop=True)
        selected, indices = catalog_test_rows(test, catalog)
        self.assertEqual(selected.product_id.tolist(), ["p1", "p3"])
        self.assertEqual(indices, [2, 0])
        changed = catalog.copy()
        changed.loc[0, "product_name"] = "different product"
        with self.assertRaisesRegex(ValueError, "differs"):
            catalog_test_rows(test, changed)
        with self.assertRaisesRegex(ValueError, "at least two"):
            catalog_test_rows(test, self.catalog.iloc[[1]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            catalog_test_rows(test, pd.concat([catalog, catalog]))

    def test_cache_provenance_shape_and_selection(self):
        identity = {"source": "fake-model"}
        features = np.eye(12, dtype=np.float32)
        for name in ("image_embeddings.npy", "text_embeddings.npy"):
            path = self.root / name
            np.save(path, features)
            write_embedding_metadata(path, identity, self.csv, features)
        images, texts = load_cached_pair(self.root, identity, self.csv, 12, [3, 1])
        np.testing.assert_array_equal(images, features[[3, 1]])
        np.testing.assert_array_equal(texts, images)
        with self.assertRaises(ValueError):
            load_cached_pair(self.root, {"source": "wrong"}, self.csv, 12, [1, 3])
        with self.assertRaisesRegex(ValueError, "dimensions"):
            load_cached_pair(self.root, identity, self.csv, 11, [1, 3])

    def test_cached_comparison_writes_all_metrics_without_loading_models(self):
        run = self.root / "run"
        run.mkdir()
        identity = {"source": "fake-model"}
        training = {"status": "complete", "dataset_sha256": "test-hash", "base_model": "fake-model",
                    "base_model_identity": identity, "checkpoint_identity": identity,
                    "best_epoch": 1, "selection_metric": "mean recall@1"}
        (run / "training.json").write_text(json.dumps(training), encoding="utf-8")
        split = self.catalog.copy()
        split["split"] = ["test"] * 3 + ["train"] * 9
        cache = self.root / "cache"
        for label in ("pretrained", "finetuned"):
            folder = cache / label
            folder.mkdir(parents=True)
            for name in ("image_embeddings.npy", "text_embeddings.npy"):
                features = np.eye(12, dtype=np.float32)
                path = folder / name
                np.save(path, features)
                write_embedding_metadata(path, identity, self.csv, features)
        output = self.root / "report"
        argv = ["compare", "--run-dir", str(run), "--catalog-csv", str(self.csv),
                "--cache-root", str(cache), "--output-dir", str(output), "--device", "cpu"]
        with patch("sys.argv", argv), patch("comparisons.compare_clip.load_experiment_data",
                return_value=(split, {"dataset_sha256": "test-hash"})), \
                patch("comparisons.compare_clip.model_identity", return_value=identity), \
                patch("comparisons.compare_clip.load_clip", side_effect=AssertionError("No model loading")), \
                contextlib.redirect_stdout(io.StringIO()):
            compare_main()
        report = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(report["catalog_size"], 12)
        self.assertEqual(report["evaluated_test_size"], 3)
        frame = pd.read_csv(output / "comparison.csv")
        self.assertEqual(len(frame), 27)
        self.assertEqual(set(frame.metric.str.split("@").str[0]), {"precision", "recall", "f1"})
        self.assertTrue((frame.delta_percentage_points == 0).all())

    def test_command_plans_and_dry_run_do_not_execute(self):
        output = self.root / "new-output"
        arguments = ["--num-products", "4000", "--output-dir", str(output)]
        stages, run = build_commands(parse_args(arguments))
        self.assertEqual([name for name, _ in stages],
                         ["prepare", "train", "encode_pretrained", "encode_finetuned", "compare"])
        self.assertEqual(run, output / "training")
        old_run = self.root / "old-run"
        old_run.mkdir()
        (old_run / "training.json").write_text(json.dumps({"status": "complete", "base_model": "base"}))
        stages, run = build_commands(parse_args(arguments + ["--reuse-run", str(old_run)]))
        self.assertNotIn("train", [name for name, _ in stages])
        self.assertEqual(run, old_run)
        with patch("comparisons.run_pipeline.subprocess.run") as execute, contextlib.redirect_stdout(io.StringIO()):
            pipeline_main(arguments + ["--dry-run"])
        execute.assert_not_called()
        self.assertFalse(output.exists())
        env = website_environment(output, old_run, self.root)
        self.assertIn("FASHION_CATALOG_DIR=", env)
        self.assertIn((old_run / "best").as_posix(), env)

    def test_failure_is_logged_and_existing_output_not_overwritten(self):
        output = self.root / "failed"
        args = ["--num-products", "4", "--output-dir", str(output),
                "--csv", str(self.csv), "--image-dir", str(self.root)]
        with patch("comparisons.run_pipeline.subprocess.run",
                   side_effect=subprocess.CalledProcessError(1, "fake")), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
            pipeline_main(args)
        report = json.loads((output / "pipeline.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["stages"][0]["status"], "failed")
        self.assertFalse((output / "website.env").exists())
        with patch("comparisons.run_pipeline.subprocess.run") as execute, \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            pipeline_main(args)
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
