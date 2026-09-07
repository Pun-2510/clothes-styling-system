from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.api import api_router
from src.logger import log_exception
from src.recommend import FashionRecommender
from src.translator import VietnameseEnglishTranslator


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.recommender = FashionRecommender()
        app.state.model_error = None

    except Exception as error:
        app.state.recommender = None
        app.state.model_error = str(error)

        log_exception(
            f"API_MODEL_INITIALIZATION_FAILED | error={error}"
        )

    try:
        app.state.translator = VietnameseEnglishTranslator()
        app.state.translator_error = None

    except Exception as error:
        app.state.translator = None
        app.state.translator_error = str(error)

        log_exception(
            f"API_TRANSLATOR_INITIALIZATION_FAILED | error={error}"
        )

    yield

    app.state.recommender = None
    app.state.translator = None


app = FastAPI(
    title="Fashion Recommendation API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_router)
