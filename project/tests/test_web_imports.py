"""Exercise relocated FastAPI package with fake models; no downloads/training."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
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


if __name__ == "__main__":
    unittest.main()
