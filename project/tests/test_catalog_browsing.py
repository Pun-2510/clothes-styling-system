"""Verify catalog browsing without loading recommendation models."""

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.backend.routers.catalog import router


app = FastAPI()
app.include_router(router, prefix="/api")


class CatalogBrowsingTests(unittest.TestCase):
    def test_categories_and_pagination(self):
        with TestClient(app) as client:
            response = client.get("/api/catalog/categories")
            self.assertEqual(response.status_code, 200)
            categories = response.json()["categories"]
            self.assertGreater(len(categories), 0)

            caps = next(item for item in categories if item["name"] == "Caps")
            page = client.get(
                "/api/catalog/products",
                params={"category": "Caps", "page": 1, "page_size": 5},
            ).json()
            self.assertEqual(page["total"], caps["count"])
            self.assertEqual(len(page["items"]), 5)
            self.assertTrue(all("image_path" not in item for item in page["items"]))

    def test_product_detail_and_invalid_category(self):
        with TestClient(app) as client:
            item = client.get(
                "/api/catalog/products",
                params={"page_size": 1},
            ).json()["items"][0]
            detail = client.get(f'/api/catalog/products/{item["product_id"]}')
            self.assertEqual(detail.status_code, 200)
            self.assertNotIn("image_path", detail.json()["product"])
            self.assertEqual(
                client.get("/api/catalog/products", params={"category": "missing"}).status_code,
                404,
            )


if __name__ == "__main__":
    unittest.main()
