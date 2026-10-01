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


if __name__ == "__main__":
    unittest.main()
