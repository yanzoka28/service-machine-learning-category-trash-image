from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.exceptions.custom_exceptions import ImageValidationError


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(image_bytes))
        image.verify()
        image = Image.open(BytesIO(image_bytes))
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("Arquivo corrompido ou não é uma imagem válida") from exc


def pil_to_cv2_rgb(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def pil_to_model_array(image: Image.Image, width: int, height: int) -> np.ndarray:
    resized = image.convert("RGB").resize((width, height))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)
