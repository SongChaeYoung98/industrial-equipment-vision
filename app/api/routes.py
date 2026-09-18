from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.schemas.analysis import (
    AnalysisResponse,
    HealthResponse,
    SimilarSearchRequest,
    SimilarSearchResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    service = request.app.state.inference_service
    return HealthResponse(status="ok", model_loaded=service.is_ready)


@router.get("/ready", response_model=HealthResponse)
def ready(request: Request) -> HealthResponse:
    service = request.app.state.inference_service
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model_not_ready",
        )
    return HealthResponse(status="ready", model_loaded=True)


@router.post("/v1/analyze", response_model=AnalysisResponse)
async def analyze(
    request: Request,
    image: Annotated[UploadFile, File(...)],
    upload_id: str,
    asset_id: str | None = None,
    modality: str = "vision",
) -> AnalysisResponse:
    service = request.app.state.inference_service
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model_not_ready",
        )

    contents = await image.read()
    try:
        return service.analyze(
            contents,
            upload_id=upload_id,
            asset_id=asset_id,
            modality=modality,
            filename=image.filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.post("/v1/search/similar", response_model=SimilarSearchResponse)
async def search_similar(
    request: Request,
    payload: SimilarSearchRequest,
) -> SimilarSearchResponse:
    service = request.app.state.inference_service
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model_not_ready",
        )
    return service.search_similar(payload)

