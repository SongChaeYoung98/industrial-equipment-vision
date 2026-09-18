from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "check-lab-ai-inference"
    app_env: str = "development"
    model_path: Path = Path("models/current/model.onnx")
    labels_path: Path = Path("configs/labels.json")
    model_version: str = "unreleased"
    preprocessing_version: str = "v1"
    max_image_bytes: int = 10 * 1024 * 1024
    inference_timeout_seconds: int = 30
    unknown_threshold: float = 0.55

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()

