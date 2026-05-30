from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "reciclapp-image-verification-demo"
    app_env: str = "development"
    max_image_size_mb: int = 5
    min_image_width: int = 300
    min_image_height: int = 300
    image_model_path: str = "app/models/recyclable_model.keras"
    image_input_width: int = 224
    image_input_height: int = 224
    blur_threshold: float = 80
    brightness_threshold: float = 50
    classification_approval_threshold: float = 0.80
    classification_review_threshold: float = 0.50
    google_application_credentials: str | None = None
    use_google_safesearch: bool = False
    classifier_demo_mode: bool = True
    input_images_dir: str = "input_images"
    output_reports_dir: str = "output_reports"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("google_application_credentials", mode="before")
    @classmethod
    def empty_credentials_to_none(cls, value: str | None) -> str | None:
        if value == "":
            return None
        return value

    @property
    def max_image_size_bytes(self) -> int:
        return self.max_image_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
