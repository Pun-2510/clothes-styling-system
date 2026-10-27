import os

from fastapi import APIRouter, HTTPException, Request, status

from src.config import (
    ACTIVE_CLIP_MODEL,
    CATEGORY_BONUS,
    CATEGORY_CONFIDENCE_FLOOR,
    CATEGORY_CONFIDENCE_THRESHOLD,
    CATEGORY_TOP_PER_CLASS,
    COMBINED_ATTRIBUTE_WEIGHT,
    COMBINED_CATEGORY_WEIGHT,
    COMBINED_IMAGE_WEIGHT,
    TEXT_RETRIEVAL_IMAGE_WEIGHT,
)


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get(
    "",
    status_code=status.HTTP_200_OK,
)
async def health_check():
    return {
        "status": "ok"
    }


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
)
async def readiness_check(request: Request):
    recommender = getattr(
        request.app.state,
        "recommender",
        None,
    )

    if recommender is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "MODEL_NOT_READY",
                "message": "Recommendation model is not ready.",
                "reason": getattr(
                    request.app.state,
                    "model_error",
                    None,
                ),
            },
        )

    return {
        "status": "ready",
        "device": str(recommender.device),
        "products": len(recommender.products),
        "image_embeddings_ready": (
            recommender.embeddings is not None
        ),
        "text_embeddings_ready": (
            recommender.has_text_embeddings
        ),
        "translator_ready": (
            getattr(
                request.app.state,
                "translator",
                None,
            ) is not None
        ),
    }


@router.get(
    "/system",
    status_code=status.HTTP_200_OK,
)
async def system_status(request: Request):
    """Return safe runtime details used by the local developer dashboard."""
    recommender = getattr(request.app.state, "recommender", None)
    translator = getattr(request.app.state, "translator", None)
    model_ready = recommender is not None

    return {
        "status": "ready" if model_ready else "degraded",
        "instance": os.environ.get("BACKEND_INSTANCE", "backend"),
        "api_version": "1.0.0",
        "device": str(recommender.device) if model_ready else None,
        "products": len(recommender.products) if model_ready else 0,
        "active_model": str(ACTIVE_CLIP_MODEL),
        "model_ready": model_ready,
        "image_embeddings_ready": bool(
            model_ready and recommender.embeddings is not None
        ),
        "text_embeddings_ready": bool(
            model_ready and recommender.has_text_embeddings
        ),
        "translator_ready": translator is not None,
        "model_error": getattr(request.app.state, "model_error", None),
        "translator_error": getattr(request.app.state, "translator_error", None),
        "retrieval": {
            "text_retrieval_image_weight": TEXT_RETRIEVAL_IMAGE_WEIGHT,
            "combined_image_weight": COMBINED_IMAGE_WEIGHT,
            "combined_attribute_weight": COMBINED_ATTRIBUTE_WEIGHT,
            "combined_category_weight": COMBINED_CATEGORY_WEIGHT,
            "image_category_bonus": CATEGORY_BONUS,
            "category_confidence_floor": CATEGORY_CONFIDENCE_FLOOR,
            "category_confidence_threshold": CATEGORY_CONFIDENCE_THRESHOLD,
            "category_top_per_class": CATEGORY_TOP_PER_CLASS,
        },
    }
