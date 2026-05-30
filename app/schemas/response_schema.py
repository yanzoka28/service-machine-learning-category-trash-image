from typing import Dict, Optional

from pydantic import BaseModel


class QualityResponse(BaseModel):
    brightness_score: float
    blur_score: float
    quality_status: str
    reason: Optional[str] = None


class SafeSearchResponse(BaseModel):
    adult: str
    spoof: str
    medical: str
    violence: str
    racy: str


class ClassificationResponse(BaseModel):
    predicted_class: str
    confidence: float
    scores_by_class: Dict[str, float]


class ImageAnalysisResponse(BaseModel):
    collection_point_id: str
    status: str
    stage: str
    reason: str
    material_informado: Optional[str] = None
    material_detectado: Optional[str] = None
    confidence: Optional[float] = None
    quality: Optional[QualityResponse] = None
    safe_search: Optional[SafeSearchResponse] = None
    classification: Optional[ClassificationResponse] = None
