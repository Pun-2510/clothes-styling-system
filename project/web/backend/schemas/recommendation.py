from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class CategoryMode(str, Enum):
    NO_CATEGORY = "no_category"
    HARD_CATEGORY = "hard_category"
    SOFT_CATEGORY = "soft_category"


class TextRecommendationRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=512,
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )
    image_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )
    language: Literal["vi", "en"] = "en"


class ProductRecommendation(BaseModel):
    product_id: str
    product_name: str | None
    category: str | None
    image_reference: str | None
    text_to_image_score: float
    text_to_text_score: float | None
    final_score: float


class TextRecommendationResponse(BaseModel):
    query: str
    query_used: str
    language: Literal["vi", "en"]
    top_k: int
    image_weight: float
    elapsed_seconds: float
    items: list[ProductRecommendation]


class ImageProductRecommendation(BaseModel):
    product_id: str
    product_name: str | None
    category: str | None
    image_reference: str | None

    visual_similarity: float
    category_bonus: float | None
    final_score: float


class ImageProcessingTimes(BaseModel):
    ranking: float
    total: float


class ImageRecommendationResponse(BaseModel):
    request_id: str
    category_mode: CategoryMode
    predicted_category: str | None
    category_confidence: float | None
    processing_times: ImageProcessingTimes
    items: list[ImageProductRecommendation]
