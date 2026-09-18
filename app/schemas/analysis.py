from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "ready"]
    model_loaded: bool


class Prediction(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class SimilarCase(BaseModel):
    upload_id: str
    similarity: float = Field(ge=0.0, le=1.0)


class AnalysisResponse(BaseModel):
    upload_id: str
    asset_id: str | None = None
    modality: str
    model_version: str
    preprocessing_version: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    unknown: bool
    top_k: list[Prediction] = Field(default_factory=list)
    similar_cases: list[SimilarCase] = Field(default_factory=list)
    processing_ms: int | None = None


class SimilarSearchRequest(BaseModel):
    upload_id: str | None = None
    image_uri: str | None = None
    modality: str = "vision"
    top_k: int = Field(default=5, ge=1, le=50)


class SimilarSearchResponse(BaseModel):
    model_version: str
    results: list[SimilarCase] = Field(default_factory=list)

