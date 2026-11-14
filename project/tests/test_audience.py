"""Unit tests for metadata-only catalog audience inference."""

import unittest

import pandas as pd

from src.audit_dataset import audit_audiences, audience_audit_summary
from src.audience import (
    add_audience_context,
    infer_audience,
    normalise_audience_preference,
)


class AudienceInferenceTests(unittest.TestCase):
    def test_explicit_adult_audiences(self):
        self.assertEqual(infer_audience("Nike Men's Polo Shirt").audience, "men")
        self.assertEqual(infer_audience("Women's Pleated Skirt").audience, "women")

    def test_children_and_unisex_have_structured_groups(self):
        child = infer_audience("Boys Graphic T-shirt")
        self.assertEqual(child.audience, "kids")
        self.assertEqual(child.age_group, "kids")
        self.assertEqual(child.gender_group, "men")
        unisex_child = infer_audience("Unisex Kids Logo Cap")
        self.assertEqual(unisex_child.audience, "kids")
        self.assertEqual(unisex_child.gender_group, "unisex")
        self.assertEqual(infer_audience("Unisex Logo Cap").audience, "unisex")
        self.assertEqual(infer_audience("Shirt for men and women").audience, "unisex")

    def test_product_style_words_do_not_create_false_child_labels(self):
        result = infer_audience(
            "Enamor Women Cream and Pink Baby Doll Nightdress",
            "A soft nightdress for women.",
            "Baby Dolls",
        )
        self.assertEqual(result.audience, "women")
        self.assertEqual(result.source, "product_name")
        self.assertEqual(
            infer_audience("Jockey Cotton Boy Leg Brief", category="Briefs").audience,
            "unknown",
        )

    def test_noisy_description_words_do_not_override_missing_name_evidence(self):
        self.assertEqual(
            infer_audience(
                "Quechua Ultralight Backpack",
                "A compact backpack for kids on a weekend hike.",
                "Backpacks",
            ).audience,
            "unknown",
        )
        self.assertEqual(
            infer_audience(
                "Domyos Navy Blue Bag",
                "Made for junior champs and everyday training.",
                "Backpacks",
            ).audience,
            "unknown",
        )

    def test_missing_evidence_remains_unknown(self):
        result = infer_audience("Blue Cotton Shirt")
        self.assertEqual(result.audience, "unknown")
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.age_group, "unknown")

    def test_explicit_preference_adds_semantic_context(self):
        self.assertEqual(
            add_audience_context("blue polo shirt", "women"),
            "blue polo shirt, women's fashion product",
        )
        self.assertEqual(normalise_audience_preference(" MEN "), "men")
        with self.assertRaises(ValueError):
            normalise_audience_preference("unknown")

    def test_audience_audit_exposes_uncertain_products(self):
        audit = audit_audiences(pd.DataFrame([
            {
                "product_id": "1",
                "product_name": "Women's Pleated Skirt",
                "category": "Skirts",
                "description": "",
            },
            {
                "product_id": "2",
                "product_name": "Blue Cotton Shirt",
                "category": "Shirts",
                "description": "",
            },
        ]))
        summary = audience_audit_summary(audit)
        self.assertEqual(summary["audience_counts"]["women"], 1)
        self.assertEqual(summary["audience_counts"]["unknown"], 1)
        self.assertEqual(summary["review_required"], 1)


if __name__ == "__main__":
    unittest.main()
