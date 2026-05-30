from pathlib import Path

from PIL import ImageStat

from app.schemas.response_schema import ClassificationResponse
from app.services.keras_classifier_service import RECYCLABLE_CLASSES


KEYWORDS_BY_CLASS = {
    "PLASTICO": ["plastico", "plastico", "plastic", "garrafa", "pet"],
    "PAPEL": ["papel", "paper", "papelao", "cardboard"],
    "VIDRO": ["vidro", "glass"],
    "METAL": ["metal", "aluminio", "lata", "can"],
    "ELETRONICO": ["eletronico", "eletronica", "eletronic", "celular", "phone"],
    "ORGANICO": ["organico", "organic", "comida", "food"],
    "GRANDE_PORTE": [
        "grande_porte",
        "grande-porte",
        "grandeporte",
        "movel",
        "sofa",
        "colchao",
        "geladeira",
        "fogao",
        "entulho",
    ],
    "LIXO_GERAL": ["lixo_geral", "lixo-geral", "lixogeral", "trash", "garbage", "rejeito"],
    "FORA_DE_CONTEXTO": ["fora", "contexto", "out"],
}


class DemoClassifierService:

    def classify(self, image, image_path: str | Path) -> ClassificationResponse:
        predicted_class = self._class_from_filename(Path(image_path).name)
        confidence = 0.88

        if predicted_class is None:
            predicted_class = self._class_from_basic_color(image)
            confidence = 0.62

        scores = self._scores(predicted_class, confidence)
        return ClassificationResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            scores_by_class=scores,
        )

    @staticmethod
    def _class_from_filename(filename: str) -> str | None:
        normalized = (
            filename.lower()
            .replace("á", "a")
            .replace("à", "a")
            .replace("ã", "a")
            .replace("â", "a")
            .replace("é", "e")
            .replace("ê", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ô", "o")
            .replace("õ", "o")
            .replace("ú", "u")
            .replace("ç", "c")
        )
        for class_name, keywords in KEYWORDS_BY_CLASS.items():
            if any(keyword in normalized for keyword in keywords):
                return class_name
        return None

    @staticmethod
    def _class_from_basic_color(image) -> str:
        mean_r, mean_g, mean_b = ImageStat.Stat(image.convert("RGB")).mean
        if mean_g > mean_r * 1.15 and mean_g > mean_b * 1.15:
            return "VIDRO"
        if mean_b > mean_r * 1.1:
            return "PLASTICO"
        if mean_r > 150 and mean_g > 130 and mean_b < 120:
            return "PAPEL"
        return "FORA_DE_CONTEXTO"

    @staticmethod
    def _scores(predicted_class: str, confidence: float) -> dict[str, float]:
        remaining = round((1 - confidence) / (len(RECYCLABLE_CLASSES) - 1), 6)
        scores = {class_name: remaining for class_name in RECYCLABLE_CLASSES}
        scores[predicted_class] = confidence
        return scores
