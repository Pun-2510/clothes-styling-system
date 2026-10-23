from fastapi import APIRouter

# Personlized imports
from .routers.health import router as health_router
from .routers.recomendations import router as recommendations_router
from .routers.catalog import router as catalog_router

api_router = APIRouter(
    prefix="/api",
)

api_router.include_router(health_router)
api_router.include_router(recommendations_router)
api_router.include_router(catalog_router)
