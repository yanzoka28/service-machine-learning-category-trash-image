class ImageVerificationError(Exception):
    def __init__(self, message: str, stage: str = "TECHNICAL_VALIDATION"):
        self.message = message
        self.stage = stage
        super().__init__(message)


class ImageValidationError(ImageVerificationError):
    pass


class SafeSearchServiceError(ImageVerificationError):
    def __init__(self, message: str):
        super().__init__(message, stage="SAFE_SEARCH_MODERATION")


class ClassifierServiceError(ImageVerificationError):
    def __init__(self, message: str):
        super().__init__(message, stage="RECYCLABLE_CLASSIFICATION")
