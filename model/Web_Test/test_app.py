import json
import threading
import unittest
import urllib.request

import numpy as np
from PIL import Image

import app


class RestoredSearchTests(unittest.TestCase):
    def test_vietnamese_fashion_query_adds_domain_aliases(self):
        normalized = app.normalize_fashion_query(
            "áo sơ mi xanh dương công sở dành cho nam"
        )

        for value in ("shirt", "blue", "office formal", "men"):
            self.assertIn(value, normalized)
        self.assertEqual(
            app.normalize_fashion_query("made in Việt Nam"),
            "made in Việt Nam",
        )

    def test_catalog_text_has_explicit_attribute_fields(self):
        text = app.catalog_product_text({
            "productDisplayName": "Blue Oxford Shirt",
            "articleType": "Shirts",
            "baseColour": "Blue",
            "gender": "Men",
        })

        self.assertIn("Product: Blue Oxford Shirt", text)
        self.assertIn("Category: Shirts", text)
        self.assertIn("Colour: Blue", text)

    def test_resnet_views_apply_consistent_flip_and_pattern_policy(self):
        image = Image.new("RGB", (100, 200), "blue")

        self.assertEqual(len(app.retrieval_views(image, False, False)), 1)
        self.assertEqual(len(app.retrieval_views(image, False, True)), 2)
        self.assertEqual(len(app.retrieval_views(image, True, True)), 8)

    def test_text_mode_ranks_catalog_without_loading_image_model(self):
        engine = app.RecommendationEngine()
        engine.products = [
            {"productDisplayName": "Blue shirt", "articleType": "Shirts"},
            {"productDisplayName": "Black jeans", "articleType": "Jeans"},
        ]
        engine.categories = ["Jeans", "Shirts"]
        engine.text_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        engine.catalog = lambda: None
        engine.load_text = lambda: None
        engine.query_embedding = lambda query: np.asarray([1.0, 0.0], dtype=np.float32)

        result = engine.search(query="blue office shirt", mode="text", top_k=1)

        self.assertEqual(result["mode"], "text")
        self.assertEqual(result["results"][0]["name"], "Blue shirt")
        self.assertIsNone(engine.image_model)

    def test_search_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "text, image hoặc combined"):
            app.RecommendationEngine().search(mode="disabled")

    def test_weighted_rrf_uses_configurable_smoothing(self):
        rankings = [np.asarray([0, 1]), np.asarray([1, 0])]
        scores = app.fused_rank_scores(rankings, (0.8, 0.2), size=2, rrf_k=10)

        self.assertGreater(scores[0], scores[1])
        with self.assertRaisesRegex(ValueError, "WEB_RRF_K"):
            app.fused_rank_scores(rankings, (0.5, 0.5), size=2, rrf_k=0)

    def test_all_three_search_tabs_are_enabled(self):
        for mode in app.SEARCH_MODES:
            marker = f'data-mode="{mode}"'
            start = app.PAGE.index(marker)
            button = app.PAGE[app.PAGE.rfind("<button", 0, start):app.PAGE.index("</button>", start)]
            self.assertNotIn("hidden", button)
            self.assertNotIn("disabled", button)

    def test_live_http_server_routes_text_search(self):
        calls = []

        class FakeEngine:
            status = "ready"
            products = [{"productDisplayName": "Blue shirt"}]
            text_model = object()
            image_model = object()

            @staticmethod
            def search(**kwargs):
                calls.append(kwargs)
                return {
                    "results": [], "metadata": {}, "mode": kwargs["mode"],
                    "catalog_size": 1, "categories": [], "cache_hits": {},
                    "device": "cpu", "query_used": kwargs["query"],
                }

        original_engine = app.ENGINE
        server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        app.ENGINE = FakeEngine()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            with urllib.request.urlopen(base + "/api/status") as response:
                status = json.load(response)
            request = urllib.request.Request(
                base + "/api/search",
                data=json.dumps({
                    "mode": "text", "query": "áo sơ mi xanh", "top_k": 6,
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request) as response:
                result = json.load(response)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            app.ENGINE = original_engine

        self.assertTrue(status["models"]["sbert_loaded"])
        self.assertEqual(result["mode"], "text")
        self.assertEqual(calls[0]["query"], "áo sơ mi xanh")


if __name__ == "__main__":
    unittest.main()
