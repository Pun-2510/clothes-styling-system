from typing import Annotated, Literal
from io import BytesIO
import uuid

from PIL import Image, UnidentifiedImageError

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from ..dependencies import get_recommender, get_translator
from ..schemas.recommendation import (
    CategoryMode,
    CombinedRecommendationResponse,
    ImageProcessingTimes,
    ImageRecommendationResponse,
    TextRecommendationRequest,
    TextRecommendationResponse,
)
from ..serializers import (
    serialize_combined_results,
    serialize_image_results,
    serialize_text_results,
)
from src.logger import log_exception
from src.recommend import FashionRecommender
from src.config import COMBINED_IMAGE_WEIGHT, TEXT_RETRIEVAL_IMAGE_WEIGHT


router = APIRouter(
    prefix="/recommendations",
    tags=["Recommendations"],
)


RecommenderDependency = Annotated[
    FashionRecommender,
    Depends(get_recommender),
]


ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

MAX_IMAGE_SIZE = 5 * 1024 * 1024


def read_uploaded_image(file: UploadFile) -> Image.Image:
    """Validate an uploaded file and return a fully loaded RGB image."""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "code": "UNSUPPORTED_IMAGE_TYPE",
                "message": "Only JPEG, PNG and WebP images are supported.",
            },
        )

    file_content = file.file.read(MAX_IMAGE_SIZE + 1)

    if not file_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "EMPTY_IMAGE",
                "message": "Uploaded image is empty.",
            },
        )

    if len(file_content) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={
                "code": "IMAGE_TOO_LARGE",
                "message": "Image size must not exceed 5 MB.",
            },
        )

    try:
        image = Image.open(BytesIO(file_content)).convert("RGB")
        image.load()
        return image
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "INVALID_IMAGE",
                "message": "Uploaded file is not a valid image.",
            },
        ) from error


@router.post(
    "/text",
    response_model=TextRecommendationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {
            "description": "Query không hợp lệ"
        },
        503: {
            "description": "Model chưa sẵn sàng"
        },
        500: {
            "description": "Lỗi xử lý recommendation"
        },
    },
)
def recommend_by_text(
    request: Request,
    payload: TextRecommendationRequest,
    recommender: RecommenderDependency,
) -> TextRecommendationResponse:
    try:
        query = payload.query.strip()

        if not query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "EMPTY_QUERY",
                    "message": "Query must not be empty.",
                },
            )

        query_used = query
        if payload.language == "vi":
            translator = get_translator(request)
            query_used = translator.translate(query)

        image_weight = (
            TEXT_RETRIEVAL_IMAGE_WEIGHT
            if payload.image_weight is None
            else payload.image_weight
        )

        results, elapsed, _ = (
            recommender.recommend_by_text(
                text=query_used,
                top_k=payload.top_k,
                image_weight=image_weight,
            )
        )

        items = serialize_text_results(results)

        return TextRecommendationResponse(
            query=query,
            query_used=query_used,
            language=payload.language,
            top_k=len(items),
            image_weight=image_weight,
            elapsed_seconds=float(elapsed),
            items=items,
        )

    except HTTPException:
        raise

    except Exception as error:
        log_exception(
            f"API_TEXT_RECOMMENDATION_FAILED | error={error}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "TEXT_RECOMMENDATION_FAILED",
                "message": "Text recommendation failed.",
            },
        ) from error


@router.post(
    "/image",
    response_model=ImageRecommendationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "File rỗng hoặc ảnh bị hỏng",
        },
        status.HTTP_413_CONTENT_TOO_LARGE: {
            "description": "File ảnh vượt quá dung lượng cho phép",
        },
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
            "description": "Định dạng file không được hỗ trợ",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Recommendation model chưa sẵn sàng",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Image recommendation thất bại",
        },
    },
)
def recommend_by_image(
    file: Annotated[
        UploadFile,
        File(description="Ảnh sản phẩm JPG, PNG hoặc WebP"),
    ],
    recommender: RecommenderDependency,
    top_k: Annotated[
        int,
        Form(ge=1, le=10),
    ] = 5,
    category_mode: Annotated[
        CategoryMode,
        Form(),
    ] = CategoryMode.NO_CATEGORY,
) -> ImageRecommendationResponse:
    image = read_uploaded_image(file)

    request_id = uuid.uuid4().hex[:8]

    try:
        recommendation = recommender.recommend_by_image(
            image=image,
            category_mode=category_mode.value,
            top_k=top_k,
            request_id=request_id,
        )

        return ImageRecommendationResponse(
            request_id=request_id,
            category_mode=CategoryMode(
                recommendation["category_mode"]
            ),
            predicted_category=recommendation[
                "predicted_category"
            ],
            category_confidence=(
                None
                if recommendation["category_confidence"] is None
                else float(
                    recommendation["category_confidence"]
                )
            ),
            processing_times=ImageProcessingTimes(
                ranking=float(
                    recommendation["ranking_time"]
                ),
                total=float(
                    recommendation["total_time"]
                ),
            ),
            items=serialize_image_results(
                recommendation["results"]
            ),
        )

    except Exception as error:
        log_exception(
            f"request={request_id} | "
            f"API_IMAGE_RECOMMENDATION_FAILED | "
            f"error={error}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "IMAGE_RECOMMENDATION_FAILED",
                "message": "Image recommendation failed.",
            },
        ) from error


@router.post(
    "/combined",
    response_model=CombinedRecommendationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Image or text query is invalid"},
        413: {"description": "Uploaded image is too large"},
        415: {"description": "Unsupported image format"},
        500: {"description": "Combined recommendation failed"},
    },
)
def recommend_by_image_and_text(
    request: Request,
    file: Annotated[
        UploadFile,
        File(description="Reference product image (JPG, PNG or WebP)"),
    ],
    recommender: RecommenderDependency,
    query: Annotated[str, Form(min_length=1, max_length=512)],
    language: Annotated[Literal["vi", "en"], Form()] = "vi",
    top_k: Annotated[int, Form(ge=1, le=10)] = 5,
    image_weight: Annotated[
        float,
        Form(ge=0.0, le=1.0),
    ] = COMBINED_IMAGE_WEIGHT,
) -> CombinedRecommendationResponse:
    query = query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "EMPTY_QUERY",
                "message": "Query must not be empty.",
            },
        )

    image = read_uploaded_image(file)
    request_id = uuid.uuid4().hex[:8]

    try:
        query_used = query
        if language == "vi":
            translator = get_translator(request)
            query_used = translator.translate(query)

        recommendation = recommender.recommend_combined(
            image=image,
            text=query_used,
            top_k=top_k,
            image_weight=image_weight,
            request_id=request_id,
        )
        items = serialize_combined_results(recommendation["results"])

        return CombinedRecommendationResponse(
            request_id=request_id,
            query=query,
            query_used=query_used,
            language=language,
            top_k=len(items),
            image_weight=float(recommendation["image_weight"]),
            text_weight=float(recommendation["text_weight"]),
            elapsed_seconds=float(recommendation["elapsed"]),
            items=items,
        )

    except HTTPException:
        raise

    except Exception as error:
        log_exception(
            f"request={request_id} | "
            f"API_COMBINED_RECOMMENDATION_FAILED | error={error}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "COMBINED_RECOMMENDATION_FAILED",
                "message": "Combined image and text recommendation failed.",
            },
        ) from error
