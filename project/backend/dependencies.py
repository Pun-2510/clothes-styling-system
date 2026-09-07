from fastapi import HTTPException, Request, status

from src.recommend import FashionRecommender
from src.translator import VietnameseEnglishTranslator


def get_recommender(
    request: Request,
) -> FashionRecommender:
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
            },
        )

    return recommender


def get_translator(
    request: Request,
) -> VietnameseEnglishTranslator:
    translator = getattr(
        request.app.state,
        "translator",
        None,
    )

    if translator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "TRANSLATOR_NOT_READY",
                "message": "Vietnamese translator is not ready.",
                "reason": getattr(
                    request.app.state,
                    "translator_error",
                    None,
                ),
            },
        )

    return translator
