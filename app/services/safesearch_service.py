import logging
from typing import Dict

from app.exceptions.custom_exceptions import SafeSearchServiceError
from app.schemas.response_schema import SafeSearchResponse

logger = logging.getLogger(__name__)

SAFE_SEARCH_LEVELS = {
    0: "UNKNOWN",
    1: "VERY_UNLIKELY",
    2: "UNLIKELY",
    3: "POSSIBLE",
    4: "LIKELY",
    5: "VERY_LIKELY",
}


class SafeSearchService:
    def __init__(self, client=None):
        self._client = client

    @property
    def client(self):
        if self._client is None:
            try:
                from google.cloud import vision

                self._client = vision.ImageAnnotatorClient()
            except Exception as exc:
                logger.exception("safe_search_client_initialization_failed")
                raise SafeSearchServiceError(
                    "Não foi possível inicializar o Google Cloud Vision"
                ) from exc
        return self._client

    def analyze(self, image_bytes: bytes) -> SafeSearchResponse:
        try:
            if self._client is None:
                from google.cloud import vision

                image = vision.Image(content=image_bytes)
            else:
                image = {"content": image_bytes}
            response = self.client.safe_search_detection(image=image)
            if getattr(response, "error", None) and response.error.message:
                raise SafeSearchServiceError("Erro ao consultar o Google Cloud Vision")

            annotation = response.safe_search_annotation
            payload: Dict[str, str] = {
                "adult": self._likelihood_name(annotation.adult),
                "spoof": self._likelihood_name(annotation.spoof),
                "medical": self._likelihood_name(annotation.medical),
                "violence": self._likelihood_name(annotation.violence),
                "racy": self._likelihood_name(annotation.racy),
            }
            return SafeSearchResponse(**payload)
        except SafeSearchServiceError:
            raise
        except Exception as exc:
            logger.exception("safe_search_analysis_failed")
            raise SafeSearchServiceError("Falha na moderação da imagem") from exc

    @staticmethod
    def _likelihood_name(value: int) -> str:
        return SAFE_SEARCH_LEVELS.get(int(value), "UNKNOWN")
