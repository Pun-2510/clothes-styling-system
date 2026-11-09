from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class CategoryMode(str, Enum):
    NO_CATEGORY = "no_category"
    HARD_CATEGORY = "hard_category"
    SOFT_CATEGORY = "soft_category"


class CategoryEvidenceCandidate(BaseModel):
    category: str
    score: float | None = None


class ExplanationGraphNode(BaseModel):
    id: str
    label: str
    role: Literal["query", "predicted", "candidate", "related", "result"]
    score: float | None = None


class ExplanationGraphEdge(BaseModel):
    source: str
    target: str
    relation: Literal["evidence", "explicit", "mutual_neighbor", "result"]
    label: str | None = None


class ExplanationGraph(BaseModel):
    nodes: list[ExplanationGraphNode] = Field(default_factory=list)
    edges: list[ExplanationGraphEdge] = Field(default_factory=list)


class RankingExplanation(BaseModel):
    query_mode: Literal["text", "image", "combined"]
    ranking_policy: str
    formula: str
    applied_category: str | None = None
    category_source: Literal["text", "image"] | None = None
    category_route: str
    category_strength: float = 0.0
    weights: dict[str, float] = Field(default_factory=dict)
    category_candidates: list[CategoryEvidenceCandidate] = Field(
        default_factory=list
    )
    graph: ExplanationGraph


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


class ProductRecommendation(BaseModel):
    product_id: str
    product_name: str | None
    category: str | None
    image_reference: str | None
    text_to_image_score: float
    text_to_text_score: float | None
    final_score: float
    reason_codes: list[str] = Field(default_factory=list)


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
    elapsed_seconds: float
    guidance: TextQueryGuidance
    items: list[ProductRecommendation]
    explanation: RankingExplanation


class ImageProductRecommendation(BaseModel):
    product_id: str
    product_name: str | None
    category: str | None
    image_reference: str | None

    visual_similarity: float
    category_bonus: float | None
    final_score: float
    reason_codes: list[str] = Field(default_factory=list)


class ImageProcessingTimes(BaseModel):
    ranking: float
    total: float


class ImageRecommendationResponse(BaseModel):
    request_id: str
    category_mode: CategoryMode
    predicted_category: str | None
    category_confidence: float | None
    candidate_categories: list[str] = Field(default_factory=list)
    processing_times: ImageProcessingTimes
    items: list[ImageProductRecommendation]
    explanation: RankingExplanation


class CombinedProductRecommendation(BaseModel):
    product_id: str
    product_name: str | None
    category: str | None
    image_reference: str | None
    visual_similarity: float
    text_to_image_score: float
    text_to_text_score: float | None
    text_score: float
    attribute_match: float | None = None
    category_match: float | None = None
    final_score: float
    reason_codes: list[str] = Field(default_factory=list)


class CombinedRecommendationResponse(BaseModel):
    request_id: str
    query: str
    query_used: str
    language: Literal["vi", "en"]
    top_k: int
    image_weight: float
    text_weight: float
    elapsed_seconds: float
    guidance: TextQueryGuidance
    items: list[CombinedProductRecommendation]
    explanation: RankingExplanation
