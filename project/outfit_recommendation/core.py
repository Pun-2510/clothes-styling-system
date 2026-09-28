"""Runtime components for the standalone outfit-recommendation prototype."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


class OutfitIndex:
    """Load a prepared Polyvore index and rank complete outfits."""

    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir).resolve()
        manifest_path = self.artifact_dir / "manifest.json"
        catalog_path = self.artifact_dir / "catalog.json"
        embeddings_path = self.artifact_dir / "item_embeddings.npy"

        missing = [
            path.name
            for path in (manifest_path, catalog_path, embeddings_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Outfit artifacts are incomplete in {self.artifact_dir}. "
                f"Missing: {', '.join(missing)}. Run prepare_index first."
            )

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        self.items: list[dict[str, Any]] = catalog["items"]
        self.outfits: list[dict[str, Any]] = catalog["outfits"]
        self.embeddings = np.load(embeddings_path, mmap_mode="r")

        if self.embeddings.ndim != 2:
            raise ValueError("item_embeddings.npy must be a two-dimensional array.")
        if len(self.items) != len(self.embeddings):
            raise ValueError(
                "Catalog and embedding counts differ; rebuild the outfit artifacts."
            )

        self.item_to_index = {
            str(item["item_id"]): index for index, item in enumerate(self.items)
        }
        if len(self.item_to_index) != len(self.items):
            raise ValueError("The prepared catalog contains duplicate item IDs.")

        expected_shape = self.manifest.get("embedding_shape")
        if expected_shape and list(self.embeddings.shape) != list(expected_shape):
            raise ValueError(
                "Embedding shape does not match manifest; rebuild the outfit artifacts."
            )

    @property
    def model_source(self) -> str:
        return str(self.manifest["model_source"])

    @property
    def embedding_dimension(self) -> int:
        return int(self.embeddings.shape[1])

    def image_path(self, item: dict[str, Any]) -> Path:
        return self.artifact_dir / str(item["image_path"])

    def rank(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        min_items: int = 3,
    ) -> list[dict[str, Any]]:
        """Rank whole outfits by their best matching anchor item.

        Similarity is used only to locate an anchor for the query. The returned
        unit is an annotated Polyvore outfit, not a list of visually similar
        products.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if min_items < 2:
            raise ValueError("min_items must be at least 2.")

        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.embedding_dimension:
            raise ValueError(
                f"Expected a {self.embedding_dimension}-dimension query, "
                f"received {query.shape[0]}."
            )
        norm = float(np.linalg.norm(query))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("The query embedding must have a finite, non-zero norm.")
        query /= norm

        item_scores = np.asarray(self.embeddings @ query, dtype=np.float32)
        ranked: list[dict[str, Any]] = []

        for outfit in self.outfits:
            item_ids = [
                str(item_id)
                for item_id in outfit["item_ids"]
                if str(item_id) in self.item_to_index
            ]
            if len(item_ids) < min_items:
                continue

            indices = np.asarray(
                [self.item_to_index[item_id] for item_id in item_ids], dtype=np.int64
            )
            local_scores = item_scores[indices]
            anchor_position = int(np.argmax(local_scores))
            anchor_id = item_ids[anchor_position]
            ranked.append(
                {
                    "outfit_id": str(outfit["outfit_id"]),
                    "score": float(local_scores[anchor_position]),
                    "matched_item_id": anchor_id,
                    "items": [self.items[index] for index in indices.tolist()],
                }
            )

        ranked.sort(key=lambda result: result["score"], reverse=True)
        return ranked[:top_k]


class ClipQueryEncoder:
    """Encode image or text queries with the model used to build the index."""

    def __init__(self, model_source: str | Path, device: str | None = None):
        from transformers import AutoProcessor, CLIPModel

        self.model_source = str(model_source)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor = AutoProcessor.from_pretrained(self.model_source)
        self.model = CLIPModel.from_pretrained(self.model_source).to(self.device)
        self.model.eval()

    @staticmethod
    def _normalise(features: torch.Tensor) -> np.ndarray:
        features = torch.nn.functional.normalize(features, dim=-1)
        return features[0].detach().cpu().numpy().astype(np.float32)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=[image.convert("RGB")], return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            output = self.model.vision_model(
                pixel_values=inputs["pixel_values"], return_dict=True
            )
            features = self.model.visual_projection(output.pooler_output)
        return self._normalise(features)

    def encode_text(self, text: str) -> np.ndarray:
        query = text.strip()
        if not query:
            raise ValueError("Text query cannot be empty.")
        inputs = self.processor(
            text=[query],
            padding=True,
            truncation=True,
            max_length=self.model.config.text_config.max_position_embeddings,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            output = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                return_dict=True,
            )
            features = self.model.text_projection(output.pooler_output)
        return self._normalise(features)
