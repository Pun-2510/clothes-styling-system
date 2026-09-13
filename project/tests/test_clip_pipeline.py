"""Offline checks: known retrieval rankings, split leakage, real CLIP gradients/CLI."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTokenizer
from tokenizers.pre_tokenizers import ByteLevel

from src.clip_data import prepare_splits, load_experiment_data
from src.clip_runtime import (model_identity, validate_embedding_metadata, write_embedding_metadata)
from src.evaluation import evaluate_retrieval, recall_at_k
from src.finetune_clip import configure_trainable, contrastive_loss, main as train_main
from src.compare_clip import main as compare_main
from src.generate_clip_embeddings import main as embeddings_main


class RecallTests(unittest.TestCase):
    def test_recall_denominator_and_cutoffs(self):
        self.assertEqual(recall_at_k([1, 2, 2], [1, 2, 3, 4], 3), 0.5)
        self.assertEqual(recall_at_k([1], [1], -1), 0)
        self.assertEqual(recall_at_k([1], [], 10), 0)

    def test_known_rankings_self_exclusion_and_singleton(self):
        products = pd.DataFrame({"product_id": ["a", "b", "c"], "category": ["shoe", "shoe", "hat"]})
        features = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 0, 1]])
        report = evaluate_retrieval(features, features, products, ks=[1, 5])
        self.assertEqual(report["text_to_image"]["recall@1"], 1)
        self.assertEqual(report["image_to_image_category"]["recall@1"], 1)
        self.assertEqual(report["image_to_image_category"]["skipped_queries"], 1)
        swapped = evaluate_retrieval(features, features[[2, 1, 0]], products, ks=[1])
        self.assertAlmostEqual(swapped["text_to_image"]["recall@1"], 1 / 3)
        with self.assertRaises(ValueError):
            evaluate_retrieval(features * 0, features, products)

    def test_category_recall_is_not_hit_rate(self):
        products = pd.DataFrame({"product_id": list("abcd"), "category": ["shoe"] * 4})
        report = evaluate_retrieval(np.eye(4), np.eye(4), products, ks=[1, 10])
        self.assertAlmostEqual(report["image_to_image_category"]["recall@1"], 1 / 3)
        self.assertEqual(report["image_to_image_category"]["recall@10"], 1)

    def test_multiple_images_of_query_product_are_excluded(self):
        products = pd.DataFrame({"product_id": ["a", "a", "b"], "category": ["shoe"] * 3})
        report = evaluate_retrieval(np.eye(3), np.eye(3), products, ks=[1])
        self.assertAlmostEqual(report["image_to_image_category"]["recall@1"], (1 + 1 + 0.5) / 3)

    def test_multi_positive_loss_matches_clip_when_pairs_unique(self):
        logits = torch.tensor([[4., 1.], [2., 3.]], requires_grad=True)
        expected = (torch.nn.functional.cross_entropy(logits, torch.arange(2))
                    + torch.nn.functional.cross_entropy(logits.T, torch.arange(2))) / 2
        self.assertTrue(torch.allclose(contrastive_loss(logits, torch.eye(2).bool()), expected))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rows = []
        for index in range(20):
            path = self.root / f"{index}.png"
            Image.new("RGB", (16, 16), (index * 11, 255 - index * 9, index * 5)).save(path)
            rows.append({"product_id": str(index), "product_name": f"shoe {index}",
                         "category": "shoe" if index < 10 else "hat", "image_path": str(path)})
        self.csv = self.root / "products.csv"
        pd.DataFrame(rows).to_csv(self.csv, index=False)

    def test_splits_are_repeatable_disjoint_and_deduplicated(self):
        data = pd.read_csv(self.csv)
        duplicate_path = self.root / "duplicate.png"
        duplicate_path.write_bytes(Path(data.iloc[0].image_path).read_bytes())
        duplicate = data.iloc[0].copy()
        duplicate["image_path"] = str(duplicate_path)
        duplicate["product_id"] = "copy"
        data = pd.concat([data, duplicate.to_frame().T], ignore_index=True)
        data.loc[1, "product_name"] = data.loc[0, "product_name"]
        data.to_csv(self.csv, index=False)
        first, metadata = prepare_splits(self.csv)
        second, _ = prepare_splits(self.csv)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(metadata["duplicate_images_removed"], 1)
        for column in ("product_id", "image_sha256", "product_text"):
            self.assertTrue((first.groupby(column).split.nunique() == 1).all())

    def test_provenance_rejects_wrong_model_and_changed_catalog(self):
        embeddings = np.eye(2, dtype=np.float32)
        path = self.root / "features.npy"
        np.save(path, embeddings)
        identity = {"source": "baseline"}
        write_embedding_metadata(path, identity, self.csv, embeddings)
        validate_embedding_metadata(path, identity, self.csv)
        with self.assertRaises(ValueError):
            validate_embedding_metadata(path, {"source": "finetuned"}, self.csv)
        self.csv.write_text("changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            validate_embedding_metadata(path, identity, self.csv)

    def test_local_checkpoint_identity_is_independent_of_mount_path(self):
        first = self.root / "windows-checkpoint"
        second = self.root / "container-checkpoint"
        first.mkdir()
        second.mkdir()
        for directory in (first, second):
            (directory / "config.json").write_text('{"model_type": "clip"}', encoding="utf-8")
            (directory / "model.safetensors").write_bytes(b"same model weights")

        self.assertEqual(model_identity(first), model_identity(second))

    def test_offline_training_comparison_and_catalog_encoding(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        base = self.root / "base"
        base.mkdir()
        alphabet = sorted(ByteLevel.alphabet())
        tokens = alphabet + [value + "</w>" for value in alphabet] + ["<|startoftext|>", "<|endoftext|>"]
        (base / "vocab.json").write_text(json.dumps(dict(zip(tokens, range(len(tokens))))), encoding="utf-8")
        (base / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
        tokenizer = CLIPTokenizer(vocab=dict(zip(tokens, range(len(tokens)))), merges=[],
                                  model_max_length=32)
        processor = CLIPProcessor(tokenizer=tokenizer, image_processor=CLIPImageProcessor(
            size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}))
        config = CLIPConfig(text_config={"vocab_size": len(tokens), "hidden_size": 16,
                            "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
                            "max_position_embeddings": 32, "bos_token_id": len(tokens) - 2,
                            "eos_token_id": len(tokens) - 1, "pad_token_id": len(tokens) - 1},
                            vision_config={"hidden_size": 16, "intermediate_size": 32,
                            "num_hidden_layers": 1, "num_attention_heads": 2, "image_size": 16,
                            "patch_size": 8}, projection_dim=8)
        model = CLIPModel(config)
        configure_trainable(model, "projections")
        self.assertFalse(model.vision_model.embeddings.patch_embedding.weight.requires_grad)
        self.assertTrue(model.visual_projection.weight.requires_grad)
        configure_trainable(model, "full")
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        original = model.visual_projection.weight.detach().clone()
        model.save_pretrained(base)
        processor.save_pretrained(base)
        run = self.root / "run"
        arguments = ["finetune", "--csv", str(self.csv), "--model", str(base), "--run-dir", str(run),
                     "--epochs", "1", "--batch-size", "4", "--learning-rate", "0.001", "--device", "cpu"]
        with patch("sys.argv", arguments + ["--prepare-only"]):
            train_main()
        with patch("sys.argv", arguments):
            train_main()
        trained = CLIPModel.from_pretrained(run / "best", local_files_only=True)
        self.assertFalse(torch.equal(original, trained.visual_projection.weight))
        with patch("sys.argv", ["compare", "--run-dir", str(run), "--device", "cpu"]):
            compare_main()
        report = json.loads((run / "comparison.json").read_text())
        self.assertEqual(set(report["metrics"]), {"pretrained", "finetuned"})
        self.assertEqual(len(report["comparison"]), 9)
        output = self.root / "embeddings"
        with patch("sys.argv", ["encode", "--model", str(run / "best"), "--csv", str(self.csv),
                                "--output-dir", str(output), "--device", "cpu"]):
            embeddings_main()
        for name in ("image_embeddings.npy", "text_embeddings.npy"):
            self.assertEqual(np.load(output / name).shape, (20, 8))
            validate_embedding_metadata(output / name, model_identity(run / "best"), self.csv)
        from src.recommend import FashionRecommender

        with patch("src.recommend.PROCESSED_CSV", self.csv), patch("src.recommend.log_info"):
            recommender = FashionRecommender(model_source=run / "best", embedding_dir=output)
            results, _, query = recommender.recommend_by_text("shoe", top_k=3)
            self.assertEqual(len(results), 3)
            self.assertEqual(query.shape, (8,))
            image_result = recommender.recommend_by_image(str(self.root / "0.png"), top_k=3)
            self.assertEqual(len(image_result["results"]), 3)
            with self.assertRaises(ValueError):
                FashionRecommender(model_source=base, embedding_dir=output)
        dataset, _ = load_experiment_data(run)
        image_path = Path(dataset.iloc[0].image_path)
        image_path.write_bytes(b"changed image")
        with self.assertRaises(ValueError):
            load_experiment_data(run)


if __name__ == "__main__":
    unittest.main()
