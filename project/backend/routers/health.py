from fastapi import APIRouter, HTTPException, Request, status


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
