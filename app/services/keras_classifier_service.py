import json
import logging
from pathlib import Path

import numpy as np

from app.config import Settings, get_settings
from app.exceptions.custom_exceptions import ClassifierServiceError
from app.schemas.response_schema import ClassificationResponse
from app.utils.image_utils import pil_to_model_array

logger = logging.getLogger(__name__)

RECYCLABLE_CLASSES = [
    "PLASTICO",
    "PAPEL",
    "VIDRO",
    "METAL",
    "ELETRONICO",
    "ORGANICO",
    "GRANDE_PORTE",
    "LIXO_GERAL",
    "FORA_DE_CONTEXTO",
]


class KerasClassifierService:
    _model = None

    def __init__(self, settings: Settings | None = None, model=None):
        self.settings = settings or get_settings()
        if model is not None:
            self.__class__._model = model

    @property
    def model(self):
        if self.__class__._model is None:
            model_path = Path(self.settings.image_model_path)
            if not model_path.exists():
                raise ClassifierServiceError("Modelo de classificação não encontrado")

            try:
                from tensorflow import keras

                self.__class__._model = keras.models.load_model(model_path)
            except Exception as exc:
                logger.exception("keras_model_load_failed")
                raise ClassifierServiceError("Não foi possível carregar o modelo Keras") from exc
        return self.__class__._model

    def labels(self) -> list[str]:
        labels_path = Path(self.settings.image_model_path).with_name("recyclable_labels.json")
        if labels_path.exists():
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
            if isinstance(labels, list) and labels:
                return labels
        return RECYCLABLE_CLASSES

    def classify(self, image) -> ClassificationResponse:
        try:
            labels = self.labels()
            input_array = pil_to_model_array(
                image,
                self.settings.image_input_width,
                self.settings.image_input_height,
            )
            predictions = self.model.predict(input_array, verbose=0)[0]
            scores = self._normalize_scores(predictions, expected_classes=len(labels))
            best_index = int(np.argmax(scores))
            scores_by_class = {
                class_name: round(float(score), 6)
                for class_name, score in zip(labels, scores)
            }
            return ClassificationResponse(
                predicted_class=labels[best_index],
                confidence=round(float(scores[best_index]), 6),
                scores_by_class=scores_by_class,
            )
        except ClassifierServiceError:
            raise
        except Exception as exc:
            logger.exception("keras_classification_failed")
            raise ClassifierServiceError("Falha na classificação da imagem") from exc

    @staticmethod
    def _normalize_scores(predictions, expected_classes: int) -> np.ndarray:
        scores = np.asarray(predictions, dtype=np.float64)
        if scores.shape[0] != expected_classes:
            raise ClassifierServiceError("Modelo retornou quantidade inesperada de classes")
        if np.any(scores < 0) or not np.isclose(scores.sum(), 1.0, atol=1e-3):
            exp_scores = np.exp(scores - np.max(scores))
            scores = exp_scores / exp_scores.sum()
        return scores
