"""Contracts keeping website artifacts independent from research experiments."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProjectArtifactIsolationTests(unittest.TestCase):
    def test_runtime_configuration_is_project_local(self):
        example = (ROOT / ".env.example").read_text(encoding="utf-8")
        expected = {
            "FASHION_CATALOG_DIR": "./data/processed",
            "FASHION_IMAGE_DIR": "./data/data",
            "FASHION_EMBEDDINGS_DIR": "./embeddings",
            "FASHION_CHECKPOINT_DIR": "./runs/clip_finetune/best",
            "FASHION_RETRIEVAL_WEIGHTS_FILE": "./retrieval_weights.json",
        }
        values = {}
        for line in example.splitlines():
            if line.startswith("FASHION_") and "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
        self.assertEqual(values, expected)
        self.assertNotIn("../experiments", example.replace("\\", "/"))

    def test_operational_python_does_not_import_experiments(self):
        files = [
            ROOT / "src" / "config.py",
            ROOT / "src" / "finetune_clip.py",
            ROOT / "src" / "retrieval_validation.py",
            ROOT / "src" / "tune_retrieval_weights.py",
        ]
        for path in files:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("from experiments", source, path.name)
            self.assertNotIn("import experiments", source, path.name)

    def test_developer_commands_use_project_pipeline(self):
        page = (
            ROOT / "web" / "frontend" / "src" / "pages" / "DeveloperPage.jsx"
        ).read_text(encoding="utf-8")
        self.assertIn("src.tune_retrieval_weights", page)
        self.assertIn("runs/clip_finetune/best", page)
        self.assertNotIn("experiments.model_comparison", page)


if __name__ == "__main__":
    unittest.main()
