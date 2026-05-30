from dataclasses import dataclass

from PIL import Image

from app.config import Settings, get_settings
from app.exceptions.custom_exceptions import ImageValidationError
from app.utils.image_utils import load_image_from_bytes


ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@dataclass(frozen=True)
class ValidatedImage:
    image_bytes: bytes
    image: Image.Image
    content_type: str


class ImageValidator:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def validate(self, image_bytes: bytes, content_type: str | None) -> ValidatedImage:
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise ImageValidationError("Formato de imagem inválido")

        if len(image_bytes) > self.settings.max_image_size_bytes:
            raise ImageValidationError("Imagem excede o tamanho máximo permitido")

        image = load_image_from_bytes(image_bytes)
        width, height = image.size
        if width < self.settings.min_image_width or height < self.settings.min_image_height:
            raise ImageValidationError("Resolução da imagem abaixo do mínimo configurado")

        return ValidatedImage(image_bytes=image_bytes, image=image, content_type=content_type)
