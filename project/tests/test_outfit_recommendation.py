import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from outfit_recommendation.core import OutfitIndex


class OutfitIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        items = [
            {"item_id": "top", "image_path": "images/top.jpg"},
            {"item_id": "pants", "image_path": "images/pants.jpg"},
            {"item_id": "shoe", "image_path": "images/shoe.jpg"},
            {"item_id": "dress", "image_path": "images/dress.jpg"},
            {"item_id": "bag", "image_path": "images/bag.jpg"},
            {"item_id": "heel", "image_path": "images/heel.jpg"},
        ]
        outfits = [
            {"outfit_id": "casual", "item_ids": ["top", "pants", "shoe"]},
            {"outfit_id": "formal", "item_ids": ["dress", "bag", "heel"]},
        ]
        embeddings = np.asarray(
            [[1, 0], [0.8, 0.2], [0.7, 0.3], [0, 1], [0.2, 0.8], [0.3, 0.7]],
            dtype=np.float32,
        )
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        np.save(self.root / "item_embeddings.npy", embeddings)
        (self.root / "catalog.json").write_text(
            json.dumps({"items": items, "outfits": outfits}), encoding="utf-8"
        )
        (self.root / "manifest.json").write_text(
            json.dumps(
                {
                    "model_source": "test-model",
                    "embedding_shape": list(embeddings.shape),
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_rank_returns_complete_outfit_for_best_anchor(self):
        index = OutfitIndex(self.root)
        results = index.rank(np.asarray([1.0, 0.0]), top_k=1)

        self.assertEqual(results[0]["outfit_id"], "casual")
        self.assertEqual(results[0]["matched_item_id"], "top")
        self.assertEqual(len(results[0]["items"]), 3)

    def test_rank_rejects_wrong_embedding_dimension(self):
        index = OutfitIndex(self.root)
        with self.assertRaisesRegex(ValueError, "Expected a 2-dimension query"):
            index.rank(np.asarray([1.0, 0.0, 0.0]))

    def test_missing_artifacts_report_prepare_step(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(FileNotFoundError, "prepare_index"):
                OutfitIndex(empty)


if __name__ == "__main__":
    unittest.main()
