from fastapi import APIRouter

# Personlized imports
from backend.routers.health import router as health_router
from backend.routers.recomendations import ( router as recommendations_router )

api_router = APIRouter(
    prefix="/api",
)

api_router.include_router(health_router)
api_router.include_router(recommendations_router)
