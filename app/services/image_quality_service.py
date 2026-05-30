import cv2

from app.config import Settings, get_settings
from app.schemas.response_schema import QualityResponse
from app.utils.image_utils import pil_to_cv2_rgb


class ImageQualityService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def analyze(self, image) -> QualityResponse:
        cv_image = pil_to_cv2_rgb(image)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        brightness_score = float(gray.mean())
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        reasons: list[str] = []
        if brightness_score < self.settings.brightness_threshold:
            reasons.append("Imagem muito escura")
        if blur_score < self.settings.blur_threshold:
            reasons.append("Imagem borrada")

        if reasons:
            return QualityResponse(
                brightness_score=round(brightness_score, 2),
                blur_score=round(blur_score, 2),
                quality_status="RUIM",
                reason="; ".join(reasons),
            )

        return QualityResponse(
            brightness_score=round(brightness_score, 2),
            blur_score=round(blur_score, 2),
            quality_status="BOA",
            reason="Imagem com qualidade suficiente",
        )
