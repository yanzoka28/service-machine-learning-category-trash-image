from app.config import Settings, get_settings
from app.schemas.response_schema import (
    ClassificationResponse,
    ImageAnalysisResponse,
    QualityResponse,
    SafeSearchResponse,
)


class DecisionService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def technical_failure(self, collection_point_id: str, material_type: str, reason: str):
        return self._response(
            collection_point_id=collection_point_id,
            status="REPROVADA",
            stage="TECHNICAL_VALIDATION",
            reason=reason,
            material_informado=material_type,
        )

    def quality_decision(self, collection_point_id: str, material_type: str, quality: QualityResponse):
        if quality.quality_status != "BOA":
            return self._response(
                collection_point_id=collection_point_id,
                status="SOLICITAR_NOVA_IMAGEM",
                stage="IMAGE_QUALITY",
                reason=quality.reason or "Imagem com qualidade insuficiente",
                material_informado=material_type,
                quality=quality,
            )
        return None

    def safe_search_decision(
        self,
        collection_point_id: str,
        material_type: str,
        safe_search: SafeSearchResponse,
        quality: QualityResponse | None = None,
    ):
        severe_levels = {"LIKELY", "VERY_LIKELY"}
        possible_levels = {"POSSIBLE"}

        if (
            safe_search.adult in severe_levels
            or safe_search.violence in severe_levels
            or safe_search.racy in severe_levels
        ):
            return self._response(
                collection_point_id=collection_point_id,
                status="REPROVADA",
                stage="SAFE_SEARCH_MODERATION",
                reason="Imagem classificada como conteúdo impróprio pelo SafeSearch",
                material_informado=material_type,
                quality=quality,
                safe_search=safe_search,
            )

        if (
            safe_search.adult in possible_levels
            or safe_search.violence in possible_levels
            or safe_search.racy in possible_levels
        ):
            return self._response(
                collection_point_id=collection_point_id,
                status="REVISAO_MANUAL",
                stage="SAFE_SEARCH_MODERATION",
                reason="Imagem considerada suspeita pelo SafeSearch",
                material_informado=material_type,
                quality=quality,
                safe_search=safe_search,
            )

        if safe_search.medical in {"POSSIBLE", "LIKELY", "VERY_LIKELY"}:
            return self._response(
                collection_point_id=collection_point_id,
                status="REVISAO_MANUAL",
                stage="SAFE_SEARCH_MODERATION",
                reason="Imagem contém indícios médicos ou sensíveis",
                material_informado=material_type,
                quality=quality,
                safe_search=safe_search,
            )

        if safe_search.spoof in severe_levels:
            return self._response(
                collection_point_id=collection_point_id,
                status="REVISAO_MANUAL",
                stage="SAFE_SEARCH_MODERATION",
                reason="Imagem possivelmente adulterada pelo SafeSearch",
                material_informado=material_type,
                quality=quality,
                safe_search=safe_search,
            )

        return None

    def final_decision(
        self,
        collection_point_id: str,
        material_type: str,
        quality: QualityResponse,
        safe_search: SafeSearchResponse,
        classification: ClassificationResponse,
    ) -> ImageAnalysisResponse:
        if classification.predicted_class == "FORA_DE_CONTEXTO":
            return self._response(
                collection_point_id,
                "REPROVADA",
                "RECYCLABLE_CLASSIFICATION",
                "Imagem fora do contexto de materiais recicláveis",
                material_type,
                classification.predicted_class,
                classification.confidence,
                quality,
                safe_search,
                classification,
            )

        if classification.confidence < self.settings.classification_review_threshold:
            return self._response(
                collection_point_id,
                "REVISAO_MANUAL",
                "FINAL_DECISION",
                "Confiança da classificação abaixo do mínimo",
                material_type,
                classification.predicted_class,
                classification.confidence,
                quality,
                safe_search,
                classification,
            )

        if classification.confidence < self.settings.classification_approval_threshold:
            return self._response(
                collection_point_id,
                "REVISAO_MANUAL",
                "FINAL_DECISION",
                "Confiança da classificação exige revisão manual",
                material_type,
                classification.predicted_class,
                classification.confidence,
                quality,
                safe_search,
                classification,
            )

        if classification.predicted_class != material_type:
            return self._response(
                collection_point_id,
                "REVISAO_MANUAL",
                "FINAL_DECISION",
                "Material detectado diferente do material informado",
                material_type,
                classification.predicted_class,
                classification.confidence,
                quality,
                safe_search,
                classification,
            )

        return self._response(
            collection_point_id,
            "APROVADA",
            "FINAL_DECISION",
            "Imagem aprovada",
            material_type,
            classification.predicted_class,
            classification.confidence,
            quality,
            safe_search,
            classification,
        )

    @staticmethod
    def _response(
        collection_point_id: str,
        status: str,
        stage: str,
        reason: str,
        material_informado: str | None = None,
        material_detectado: str | None = None,
        confidence: float | None = None,
        quality: QualityResponse | None = None,
        safe_search: SafeSearchResponse | None = None,
        classification: ClassificationResponse | None = None,
    ) -> ImageAnalysisResponse:
        return ImageAnalysisResponse(
            collection_point_id=collection_point_id,
            status=status,
            stage=stage,
            reason=reason,
            material_informado=material_informado,
            material_detectado=material_detectado,
            confidence=confidence,
            quality=quality,
            safe_search=safe_search,
            classification=classification,
        )
