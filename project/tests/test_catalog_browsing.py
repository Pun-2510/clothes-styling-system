"""Verify catalog browsing without loading recommendation models."""

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.backend.routers.catalog import _catalog, router


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
            self.assertTrue(all("audience" in item for item in page["items"]))
            self.assertTrue(all("age_group" in item for item in page["items"]))
            self.assertTrue(all("gender_group" in item for item in page["items"]))
            self.assertTrue(all("audience_evidence" in item for item in page["items"]))

    def test_audience_browsing_is_an_independent_catalog_filter(self):
        with TestClient(app) as client:
            response = client.get("/api/catalog/audiences")
            self.assertEqual(response.status_code, 200)
            audiences = response.json()["audiences"]
            self.assertEqual(
                [item["name"] for item in audiences],
                ["women", "men", "kids", "unisex", "unknown"],
            )
            catalog_total = client.get(
                "/api/catalog/products", params={"page_size": 1}
            ).json()["total"]
            self.assertEqual(sum(item["count"] for item in audiences), catalog_total)

            page = client.get(
                "/api/catalog/products",
                params={"audience": "women", "page_size": 5},
            )
            self.assertEqual(page.status_code, 200)
            self.assertTrue(page.json()["items"])
            self.assertTrue(
                all(item["audience"] == "women" for item in page.json()["items"])
            )
            self.assertEqual(
                client.get("/api/catalog/products", params={"audience": "missing"}).status_code,
                404,
            )

    def test_child_page_excludes_known_false_positive_phrases(self):
        _, _, by_audience, _ = _catalog()
        child_names = {
            str(item["product_name"]) for item in by_audience.get("kids", [])
        }
        self.assertFalse(any("Enamor Women" in name for name in child_names))
        self.assertFalse(any("Quechua Ultralight Backpack" in name for name in child_names))
        self.assertFalse(any("Domyos Navy Blue Bag" in name for name in child_names))

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
