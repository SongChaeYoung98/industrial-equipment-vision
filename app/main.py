from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.services.inference_service import InferenceService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.inference_service = InferenceService(settings)
    yield


app = FastAPI(
    title="check-lab-ai inference API",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router)

