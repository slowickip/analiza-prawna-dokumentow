"""Structure errors."""


class StructureError(Exception):
    """Raised when deterministic segmentation or internal reference parsing fails.

    Signals an unextractable document: empty text, or no qualifying
    paragraph/outline units. Callers reject the upload.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)
