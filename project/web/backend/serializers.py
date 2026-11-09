import pandas as pd

from .schemas.recommendation import (
    CombinedProductRecommendation,
    ImageProductRecommendation,
    ProductRecommendation,
)

def optional_string(value) -> str | None:
    if value is None or pd.isna(value):
        return None

    return str(value)


def serialize_text_results(
    results: pd.DataFrame,
) -> list[ProductRecommendation]:
    items = []

    for _, row in results.iterrows():
        text_score = row.get("text_similarity")

        reason_codes = ["text_image_match"]
        if text_score is not None and not pd.isna(text_score):
            reason_codes.append("text_description_match")

        items.append(
            ProductRecommendation(
                product_id=str(row["product_id"]),
                product_name=optional_string(
                    row.get("product_name")
                ),
                category=optional_string(
                    row.get("category")
                ),
                image_reference=optional_string(
                    row.get("image_reference")
                ),
                text_to_image_score=float(
                    row["cross_modal_similarity"]
                ),
                text_to_text_score=(
                    None
                    if text_score is None or pd.isna(text_score)
                    else float(text_score)
                ),
                final_score=float(row["similarity"]),
                reason_codes=reason_codes,
            )
        )

    return items


def optional_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None

    return float(value)


def serialize_image_results(
    results: pd.DataFrame,
) -> list[ImageProductRecommendation]:
    items = []

    for _, row in results.iterrows():
        category_bonus = optional_float(row.get("category_bonus"))
        reason_codes = ["visual_similarity"]
        if category_bonus is not None and category_bonus > 0:
            reason_codes.append("category_bonus")

        items.append(
            ImageProductRecommendation(
                product_id=str(row["product_id"]),
                product_name=optional_string(
                    row.get("product_name")
                ),
                category=optional_string(
                    row.get("category")
                ),
                image_reference=optional_string(
                    row.get("image_reference")
                ),
                visual_similarity=float(
                    row["visual_similarity"]
                ),
                category_bonus=category_bonus,
                final_score=float(
                    row["similarity"]
                ),
                reason_codes=reason_codes,
            )
        )

    return items


def serialize_combined_results(
    results: pd.DataFrame,
) -> list[CombinedProductRecommendation]:
    items = []

    for _, row in results.iterrows():
        attribute_match = optional_float(row.get("attribute_match"))
        category_match = optional_float(row.get("category_match"))
        reason_codes = ["visual_similarity", "text_query_match"]
        if attribute_match is not None and attribute_match > 0:
            reason_codes.append("attribute_match")
        if category_match is not None and category_match > 0:
            reason_codes.append("category_evidence")

        items.append(
            CombinedProductRecommendation(
                product_id=str(row["product_id"]),
                product_name=optional_string(row.get("product_name")),
                category=optional_string(row.get("category")),
                image_reference=optional_string(row.get("image_reference")),
                visual_similarity=float(row["visual_similarity"]),
                text_to_image_score=float(row["text_to_image_similarity"]),
                text_to_text_score=optional_float(
                    row.get("text_to_text_similarity")
                ),
                text_score=float(row["text_similarity"]),
                attribute_match=attribute_match,
                category_match=category_match,
                final_score=float(row["similarity"]),
                reason_codes=reason_codes,
            )
        )

    return items
