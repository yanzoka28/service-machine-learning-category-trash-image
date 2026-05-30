import mimetypes
from pathlib import Path

from app.config import Settings, get_settings
from app.exceptions.custom_exceptions import (
    ClassifierServiceError,
    ImageValidationError,
    SafeSearchServiceError,
)
from app.schemas.response_schema import ImageAnalysisResponse, SafeSearchResponse
from app.services.decision_service import DecisionService
from app.services.demo_classifier_service import DemoClassifierService
from app.services.image_quality_service import ImageQualityService
from app.services.image_validator import ImageValidator
from app.services.keras_classifier_service import KerasClassifierService
from app.services.safesearch_service import SafeSearchService


class ImageAnalysisPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.decisions = DecisionService(self.settings)

    def analyze_file(
        self,
        image_path: str | Path,
        material_type: str,
        collection_point_id: str,
    ) -> ImageAnalysisResponse:
        path = Path(image_path)
        material = material_type.upper()
        content_type = mimetypes.guess_type(path.name)[0]

        try:
            image_bytes = path.read_bytes()
            validated = ImageValidator(self.settings).validate(image_bytes, content_type)
        except FileNotFoundError:
            return self.decisions.technical_failure(
                collection_point_id,
                material,
                f"Arquivo não encontrado: {path}",
            )
        except ImageValidationError as exc:
            return self.decisions.technical_failure(collection_point_id, material, exc.message)

        quality = ImageQualityService(self.settings).analyze(validated.image)
        quality_decision = self.decisions.quality_decision(collection_point_id, material, quality)
        if quality_decision:
            return quality_decision

        safe_search = self._safe_search(validated.image_bytes)
        moderation_decision = self.decisions.safe_search_decision(
            collection_point_id,
            material,
            safe_search,
            quality,
        )
        if moderation_decision:
            return moderation_decision

        try:
            classification = KerasClassifierService(self.settings).classify(validated.image)
        except ClassifierServiceError:
            if not self.settings.classifier_demo_mode:
                raise
            classification = DemoClassifierService().classify(validated.image, path)

        return self.decisions.final_decision(
            collection_point_id,
            material,
            quality,
            safe_search,
            classification,
        )

    def _safe_search(self, image_bytes: bytes) -> SafeSearchResponse:
        if not self.settings.use_google_safesearch:
            return SafeSearchResponse(
                adult="VERY_UNLIKELY",
                spoof="UNLIKELY",
                medical="VERY_UNLIKELY",
                violence="VERY_UNLIKELY",
                racy="VERY_UNLIKELY",
            )

        try:
            return SafeSearchService().analyze(image_bytes)
        except SafeSearchServiceError:
            return SafeSearchResponse(
                adult="POSSIBLE",
                spoof="UNLIKELY",
                medical="VERY_UNLIKELY",
                violence="VERY_UNLIKELY",
                racy="VERY_UNLIKELY",
            )
