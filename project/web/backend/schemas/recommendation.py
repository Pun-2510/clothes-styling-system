from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

AudiencePreference = Literal["women", "men", "kids", "unisex"]

class CategoryMode(str, Enum):
    NO_CATEGORY = "no_category"
    HARD_CATEGORY = "hard_category"


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
    image_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    language: Literal["vi", "en"] = "en"
    audience: AudiencePreference | None = None


class ProductRecommendation(BaseModel):
    product_id: str
    product_name: str | None
    category: str | None
    image_reference: str | None
    audience: str | None = None
    text_to_image_score: float
    text_to_text_score: float | None
    metadata_match: float | None = None
    final_score: float


class CategorySuggestion(BaseModel):
    category: str
    corrected_query: str
    confidence: float | None = None


class TextQueryGuidance(BaseModel):
    status: Literal[
        "category_applied",
        "spelling_suggestion",
        "broad_query",
        "multiple_categories",
        "unrestricted",
    ]
    applied_category: str | None = None
    suggestions: list[CategorySuggestion] = Field(default_factory=list)


class TextRecommendationResponse(BaseModel):
    query: str
    query_used: str
    language: Literal["vi", "en"]
    top_k: int
    image_weight: float
    audience: AudiencePreference | None = None
    elapsed_seconds: float
    guidance: TextQueryGuidance
    items: list[ProductRecommendation]


class ImageProductRecommendation(BaseModel):
    product_id: str
    product_name: str | None
    category: str | None
    image_reference: str | None
    audience: str | None = None

    visual_similarity: float
    final_score: float


class ImageProcessingTimes(BaseModel):
    ranking: float
    total: float


class ImageRecommendationResponse(BaseModel):
    request_id: str
    category_mode: CategoryMode
    predicted_category: str | None
    category_confidence: float | None
    candidate_categories: list[str] = Field(default_factory=list)
    audience: AudiencePreference | None = None
    processing_times: ImageProcessingTimes
    items: list[ImageProductRecommendation]


class CombinedProductRecommendation(BaseModel):
    product_id: str
    product_name: str | None
    category: str | None
    image_reference: str | None
    audience: str | None = None
    visual_similarity: float
    text_to_image_score: float
    text_to_text_score: float | None
    text_score: float
    metadata_match: float | None = None
    category_match: float | None = None
    final_score: float


class CombinedRecommendationResponse(BaseModel):
    request_id: str
    query: str
    query_used: str
    language: Literal["vi", "en"]
    top_k: int
    image_weight: float
    text_weight: float
    audience: AudiencePreference | None = None
    elapsed_seconds: float
    guidance: TextQueryGuidance
    items: list[CombinedProductRecommendation]
