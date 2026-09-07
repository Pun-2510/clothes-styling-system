import pandas as pd

from backend.schemas.recommendation import (
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
                category_bonus=optional_float(
                    row.get("category_bonus")
                ),
                final_score=float(
                    row["similarity"]
                ),
            )
        )

    return items