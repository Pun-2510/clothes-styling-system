"""Exercise relocated FastAPI package with fake models; no downloads/training."""

from io import BytesIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image
import pandas as pd
from web.backend.main import app


class WebPackageTests(unittest.TestCase):
    def test_app_startup_routes_and_dependencies_after_move(self):
        recommender = SimpleNamespace(device="cpu", products=[1, 2],
                                      embeddings=object(), has_text_embeddings=True)
        with patch("web.backend.main.FashionRecommender", return_value=recommender) as model, \
                patch("web.backend.main.VietnameseEnglishTranslator", return_value=object()) as translator, \
                TestClient(app) as client:
            self.assertEqual(client.get("/api/health").status_code, 200)
            response = client.get("/api/health/ready")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["products"], 2)
            self.assertTrue(response.json()["translator_ready"])
            routes = client.get("/openapi.json").json()["paths"]
            self.assertIn("/api/recommendations/image", routes)
            self.assertIn("/api/recommendations/text", routes)
            self.assertIn("/api/recommendations/combined", routes)
            response = client.post("/api/recommendations/text", json={"query": "   "})
            self.assertEqual(response.status_code, 400)
            response = client.post("/api/recommendations/image",
                                   files={"file": ("bad.txt", b"not an image", "text/plain")})
            self.assertEqual(response.status_code, 415)
            model.assert_called_once()
            translator.assert_called_once()

    def test_image_route_reports_mode_applied_after_automatic_fallback(self):
        results = pd.DataFrame({
            "product_id": ["1"],
            "product_name": ["Product"],
            "category": ["Shirts"],
            "image_reference": ["1.jpg"],
            "visual_similarity": [0.8],
            "similarity": [0.8],
        })
        recommend_by_image = Mock(return_value={
            "category_mode": "no_category",
            "predicted_category": None,
            "category_confidence": None,
            "ranking_time": 0.01,
            "total_time": 0.02,
            "results": results,
        })
        recommender = SimpleNamespace(
            device="cpu",
            products=[1],
            embeddings=object(),
            has_text_embeddings=True,
            recommend_by_image=recommend_by_image,
        )
        image_buffer = BytesIO()
        Image.new("RGB", (2, 2), "white").save(image_buffer, format="PNG")

        with patch("web.backend.main.FashionRecommender", return_value=recommender), \
                patch("web.backend.main.VietnameseEnglishTranslator", return_value=object()), \
                TestClient(app) as client:
            response = client.post(
                "/api/recommendations/image",
                data={"top_k": "5", "category_mode": "soft_category"},
                files={"file": ("query.png", image_buffer.getvalue(), "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["category_mode"], "no_category")
        self.assertEqual(
            recommend_by_image.call_args.kwargs["category_mode"],
            "soft_category",
        )

    def test_combined_route_translates_and_serializes_fusion_scores(self):
        results = pd.DataFrame({
            "product_id": ["1"],
            "product_name": ["Black dress"],
            "category": ["Dresses"],
            "image_reference": ["1.jpg"],
            "visual_similarity": [0.9],
            "text_to_image_similarity": [0.7],
            "text_to_text_similarity": [0.8],
            "text_similarity": [0.75],
            "similarity": [0.825],
        })
        recommend_combined = Mock(return_value={
            "results": results,
            "image_weight": 0.5,
            "text_weight": 0.5,
            "elapsed": 0.03,
        })
        recommender = SimpleNamespace(
            device="cpu",
            products=[1],
            embeddings=object(),
            has_text_embeddings=True,
            recommend_combined=recommend_combined,
        )
        translator = Mock()
        translator.translate.return_value = "black office dress"
        image_buffer = BytesIO()
        Image.new("RGB", (2, 2), "black").save(image_buffer, format="PNG")

        with patch("web.backend.main.FashionRecommender", return_value=recommender), \
                patch("web.backend.main.VietnameseEnglishTranslator", return_value=translator), \
                TestClient(app) as client:
            response = client.post(
                "/api/recommendations/combined",
                data={
                    "query": "váy đen công sở",
                    "language": "vi",
                    "top_k": "5",
                    "image_weight": "0.5",
                },
                files={"file": ("query.png", image_buffer.getvalue(), "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query_used"], "black office dress")
        self.assertEqual(payload["items"][0]["text_score"], 0.75)
        self.assertEqual(
            recommend_combined.call_args.kwargs["text"],
            "black office dress",
        )


if __name__ == "__main__":
    unittest.main()
