import json
from pathlib import Path

from app.core.config import Settings
from app.schemas.analysis import (
    AnalysisResponse,
    SimilarSearchRequest,
    SimilarSearchResponse,
)


class InferenceService:
    """운영 추론 경계.

    현재는 모델 파일이 없으므로 실제 추론을 수행하지 않습니다.
    모델을 등록하면 이 클래스 안에서 전처리·ONNX Runtime·검색 인덱스를 연결합니다.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.labels = self._load_labels(settings.labels_path)
        self.is_ready = settings.model_path.is_file()
        self._session = None
        self._index = None

    @staticmethod
    def _load_labels(path: Path) -> list[str]:
        if not path.is_file():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [item["id"] for item in payload.get("labels", [])]

    def analyze(
        self,
        image_bytes: bytes,
        *,
        upload_id: str,
        asset_id: str | None,
        modality: str,
        filename: str | None,
    ) -> AnalysisResponse:
        if len(image_bytes) > self.settings.max_image_bytes:
            raise ValueError("image_too_large")
        if not image_bytes:
            raise ValueError("empty_image")

        # TODO: validate/decode image, apply the training-time preprocessing,
        # run the ONNX session, and query the embedding index.
        del filename
        raise NotImplementedError(
            "model inference is not configured; place a versioned model at "
            f"{self.settings.model_path}"
        )

    def search_similar(self, payload: SimilarSearchRequest) -> SimilarSearchResponse:
        # TODO: resolve image_uri, create an embedding, and query the index.
        del payload
        return SimilarSearchResponse(
            model_version=self.settings.model_version,
            results=[],
        )
